import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from easydict import EasyDict as CN

# ========================================
# 1. Swin Transformer 配置
# ========================================
def swin_config():
    _C = CN()
    _C.DATA = CN()
    _C.DATA.IMG_SIZE = 224

    _C.MODEL = CN()
    _C.MODEL.NUM_CLASSES = 1
    _C.MODEL.DROP_RATE = 0.0
    _C.MODEL.DROP_PATH_RATE = 0.3

    _C.MODEL.SWIN = CN()
    _C.MODEL.SWIN.PATCH_SIZE = 4
    _C.MODEL.SWIN.IN_CHANS = 3
    _C.MODEL.SWIN.EMBED_DIM = 96
    _C.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
    _C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    _C.MODEL.SWIN.WINDOW_SIZE = 7
    _C.MODEL.SWIN.MLP_RATIO = 4.0
    _C.MODEL.SWIN.QKV_BIAS = True
    _C.MODEL.SWIN.APE = False
    _C.MODEL.SWIN.PATCH_NORM = True

    _C.TRAIN = CN()
    _C.TRAIN.USE_CHECKPOINT = False
    return _C


# ========================================
# 2. Swin 基础模块（Mlp, WindowAttention, etc.）
# ========================================
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


# ========================================
# 3. Swin Transformer 主干（返回特征 + 预测）
# ========================================
class SwinTransformer(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super().__init__()
        self.cfg = cfg
        self.return_feature = return_feat

        self.patch_embed = PatchEmbed(
            img_size=cfg.DATA.IMG_SIZE,
            patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
            in_chans=cfg.MODEL.SWIN.IN_CHANS,
            embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
            norm_layer=nn.LayerNorm if cfg.MODEL.SWIN.PATCH_NORM else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        self.ape = cfg.MODEL.SWIN.APE
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.MODEL.SWIN.EMBED_DIM))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=cfg.MODEL.DROP_RATE)

        dpr = [x.item() for x in torch.linspace(0, cfg.MODEL.DROP_PATH_RATE, sum(cfg.MODEL.SWIN.DEPTHS))]

        self.layers = nn.ModuleList()
        for i_layer in range(len(cfg.MODEL.SWIN.DEPTHS)):
            layer = BasicLayer(
                dim=int(cfg.MODEL.SWIN.EMBED_DIM * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=cfg.MODEL.SWIN.DEPTHS[i_layer],
                num_heads=cfg.MODEL.SWIN.NUM_HEADS[i_layer],
                window_size=cfg.MODEL.SWIN.WINDOW_SIZE,
                mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                qkv_bias=cfg.MODEL.SWIN.QKV_BIAS,
                drop=cfg.MODEL.DROP_RATE,
                attn_drop=cfg.MODEL.DROP_RATE,
                drop_path=dpr[sum(cfg.MODEL.SWIN.DEPTHS[:i_layer]):sum(cfg.MODEL.SWIN.DEPTHS[:i_layer+1])],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging if (i_layer < len(cfg.MODEL.SWIN.DEPTHS) - 1) else None
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(cfg.MODEL.SWIN.EMBED_DIM * 8)  # 96 * 8 = 768
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(cfg.MODEL.SWIN.EMBED_DIM * 8, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        return torch.flatten(x, 1)

    def forward(self, x):
        feat = self.forward_features(x)
        out = self.head(feat)
        out = torch.sigmoid(out) * 5  # 输出 0~5 分
        return (out, feat) if self.return_feature else out


# ========================================
# 4. 视频时序聚合模型
# ========================================
class VideoPersonalityModel(nn.Module):
    def __init__(self, swin_model: SwinTransformer, agg_type="attention"):
        super().__init__()
        self.swin = swin_model
        self.agg_type = agg_type
        self.feat_dim = 768

        if agg_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.feat_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
        elif agg_type == "lstm":
            self.lstm = nn.LSTM(self.feat_dim, 256, 2, batch_first=True, bidirectional=True)
            self.fc_lstm = nn.Linear(512, self.feat_dim)

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        _, frame_feat = self.swin(video_frames.view(B*T, C, H, W))
        frame_feat = frame_feat.view(B, T, self.feat_dim)

        if self.agg_type == "mean":
            video_feat = frame_feat.mean(dim=1)
        elif self.agg_type == "attention":
            att_weights = F.softmax(self.attention(frame_feat), dim=1)
            video_feat = torch.sum(frame_feat * att_weights, dim=1)
        elif self.agg_type == "lstm":
            lstm_out, _ = self.lstm(frame_feat)
            video_feat = self.fc_lstm(lstm_out[:, -1, :])
        else:
            raise ValueError("agg_type must be 'mean', 'attention', or 'lstm'")

        pred = self.swin.head(self.swin.pos_drop(video_feat))
        return torch.sigmoid(pred) * 5  # 输出 0~5 分


# ========================================
# 5. 视频关键帧提取 + 数据增强
# ========================================
def extract_keyframes(video_path: str, num_frames: int = 16) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"无法打开视频: {video_path}")

    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
    cap.release()
    return frames


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ========================================
# 6. 数据集（原始标签）
# ========================================
class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str, num_frames: int = 16):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = get_train_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_dir, row['video_name'])
        frames = extract_keyframes(video_path, self.num_frames)
        frame_tensors = [self.transform(f) for f in frames]
        frames_tensor = torch.stack(frame_tensors)  # (T, 3, 224, 224)

        label = torch.tensor(row['Extraversion'], dtype=torch.float32).unsqueeze(0)  # (1,)
        return frames_tensor, label


# ========================================
# 7. 训练函数（每轮保存 + 原始 MAE）
# ========================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_mae = 0.0

    # 动态 desc，显示当前 epoch
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} | Training", leave=True)
    
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(frames)
        loss = criterion(preds, labels)
        mae = torch.mean(torch.abs(preds - labels))

        loss.backward()
        optimizer.step()

        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_mae += mae.item() * batch_size

        # 实时更新 pbar 的后缀（loss/mae）
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae.item():.4f}'
        })

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


# ========================================
# 8. 主训练流程（增强信息输出）
# ========================================
def main():
    cfg = swin_config()
    C = {
        "video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Extraversion_train",
        "data_csv": "/root/autodl-tmp/data/extraversion_train.csv",
        "num_frames": 16,
        "agg_type": "attention",
        "batch_size": 4,
        "epochs": 50,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "save_dir": "/root/autodl-tmp/swin-transformer/saved_models/Extraversion_Swin",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    os.makedirs(C["save_dir"], exist_ok=True)

    # 加载数据
    df = pd.read_csv(C["data_csv"])
    if not {'video_name', 'Extraversion'}.issubset(df.columns):
        raise ValueError("CSV 必须包含 'video_name' 和 'Extraversion' 列")

    dataset = VideoDataset(df, C["video_dir"], C["num_frames"])
    loader = DataLoader(dataset, batch_size=C["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    # 打印关键信息
    print(f"数据集加载完成，共 {len(dataset)} 个训练视频")
    print(f"Batch Size: {C['batch_size']} → 共 {len(loader)} 个 batch/epoch")
    print(f"使用设备: {C['device']}")

    # 构建模型
    swin = SwinTransformer(cfg, return_feat=True).to(C["device"])
    model = VideoPersonalityModel(swin, agg_type=C["agg_type"]).to(C["device"])

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=C["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # 训练循环
    for epoch in range(1, C["epochs"] + 1):
        loss, mae = train_one_epoch(
            model, loader, criterion, optimizer, C["device"], 
            epoch=epoch, total_epochs=C["epochs"]
        )
        print(f"\nEpoch {epoch}/{C['epochs']} | Avg Loss: {loss:.4f} | Avg MAE: {mae:.4f}")
        scheduler.step()

        # 每轮保存
        save_path = os.path.join(C["save_dir"], f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")

    print(f"\n训练完成！共训练 {C['epochs']} 轮，所有模型已保存至:\n{C['save_dir']}")


if __name__ == "__main__":
    main()