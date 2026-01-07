import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

# 设置日志
logging.basicConfig(
    filename="/root/autodl-tmp/swin-transformer/saved_models/Extraversion_Swin/test_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ========================================
# 1. Swin Transformer 配置（与训练一致）
# ========================================
from easydict import EasyDict as CN

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
# 2. Swin 基础模块（复制训练代码）
# ========================================
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

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
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
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
        self.dim = dim; self.window_size = window_size; self.num_heads = num_heads
        head_dim = dim // num_heads; self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads))
        coords_h = torch.arange(window_size[0]); coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1; relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias); self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim); self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02); self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]; q = q * self.scale; attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1], self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]; attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N); attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn); x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x); x = self.proj_drop(x); return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim; self.input_resolution = input_resolution; self.num_heads = num_heads
        self.window_size = window_size; self.shift_size = shift_size; self.mlp_ratio = mlp_ratio
        if min(input_resolution) <= window_size: self.shift_size = 0; self.window_size = min(input_resolution)
        self.norm1 = norm_layer(dim); self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim); mlp_hidden_dim = int(dim * mlp_ratio); self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution; img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, self.window_size); mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution; B, L, C = x.shape; shortcut = x; x = self.norm1(x); x = x.view(B, H, W, C)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size); x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask); attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = x.view(B, H * W, C); x = shortcut + self.drop_path(x); x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution; self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False); self.norm = norm_layer(4 * dim)
    def forward(self, x):
        H, W = self.input_resolution; B, L, C = x.shape; x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]; x1 = x[:, 1::2, 0::2, :]; x2 = x[:, 0::2, 1::2, :]; x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1); x = x.view(B, -1, 4 * C); x = self.norm(x); x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim; self.input_resolution = input_resolution; self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None
    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        if self.downsample is not None: x = self.downsample(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size); patch_size = to_2tuple(patch_size)
        self.img_size = img_size; self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans; self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
    def forward(self, x):
        B, C, H, W = x.shape; x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm: x = self.norm(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super().__init__()
        self.cfg = cfg; self.return_feature = return_feat
        self.patch_embed = PatchEmbed(img_size=cfg.DATA.IMG_SIZE, patch_size=cfg.MODEL.SWIN.PATCH_SIZE,
                                      in_chans=cfg.MODEL.SWIN.IN_CHANS, embed_dim=cfg.MODEL.SWIN.EMBED_DIM,
                                      norm_layer=nn.LayerNorm if cfg.MODEL.SWIN.PATCH_NORM else None)
        num_patches = self.patch_embed.num_patches; patches_resolution = self.patch_embed.patches_resolution
        self.ape = cfg.MODEL.SWIN.APE
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, cfg.MODEL.SWIN.EMBED_DIM))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=cfg.MODEL.DROP_RATE)
        dpr = [x.item() for x in torch.linspace(0, cfg.MODEL.DROP_PATH_RATE, sum(cfg.MODEL.SWIN.DEPTHS))]
        self.layers = nn.ModuleList()
        for i_layer in range(len(cfg.MODEL.SWIN.DEPTHS)):
            layer = BasicLayer(dim=int(cfg.MODEL.SWIN.EMBED_DIM * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                               depth=cfg.MODEL.SWIN.DEPTHS[i_layer], num_heads=cfg.MODEL.SWIN.NUM_HEADS[i_layer],
                               window_size=cfg.MODEL.SWIN.WINDOW_SIZE, mlp_ratio=cfg.MODEL.SWIN.MLP_RATIO,
                               qkv_bias=cfg.MODEL.SWIN.QKV_BIAS, drop=cfg.MODEL.DROP_RATE, attn_drop=cfg.MODEL.DROP_RATE,
                               drop_path=dpr[sum(cfg.MODEL.SWIN.DEPTHS[:i_layer]):sum(cfg.MODEL.SWIN.DEPTHS[:i_layer+1])],
                               norm_layer=nn.LayerNorm,
                               downsample=PatchMerging if (i_layer < len(cfg.MODEL.SWIN.DEPTHS) - 1) else None)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(cfg.MODEL.SWIN.EMBED_DIM * 8)  # 768
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(cfg.MODEL.SWIN.EMBED_DIM * 8, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape: x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers: x = layer(x)
        x = self.norm(x); x = self.avgpool(x.transpose(1, 2)); return torch.flatten(x, 1)

    def forward(self, x):
        feat = self.forward_features(x)
        out = self.head(feat)
        out = torch.sigmoid(out) * 5  # 0~5 分
        return (out, feat) if self.return_feature else out


# ========================================
# 3. 视频时序聚合模型（与训练一致）
# ========================================
class VideoPersonalityModel(nn.Module):
    def __init__(self, swin_model: SwinTransformer, agg_type="attention"):
        super().__init__()
        self.swin = swin_model
        self.agg_type = agg_type
        self.feat_dim = 768
        if agg_type == "attention":
            self.attention = nn.Sequential(nn.Linear(768, 256), nn.Tanh(), nn.Linear(256, 1))
        elif agg_type == "lstm":
            self.lstm = nn.LSTM(768, 256, 2, batch_first=True, bidirectional=True)
            self.fc_lstm = nn.Linear(512, 768)

    def forward(self, video_frames):
        B, T, C, H, W = video_frames.shape
        _, frame_feat = self.swin(video_frames.view(B*T, C, H, W))
        frame_feat = frame_feat.view(B, T, self.feat_dim)
        if self.agg_type == "mean":
            video_feat = frame_feat.mean(1)
        elif self.agg_type == "attention":
            w = F.softmax(self.attention(frame_feat), dim=1)
            video_feat = (frame_feat * w).sum(1)
        elif self.agg_type == "lstm":
            out, _ = self.lstm(frame_feat)
            video_feat = self.fc_lstm(out[:, -1])
        pred = self.swin.head(self.swin.pos_drop(video_feat))
        return torch.sigmoid(pred) * 5  # 0~5 分


# ========================================
# 4. 工具函数
# ========================================
def extract_keyframes(video_path: str, num_frames: int = 16) -> List[Image.Image]:
    if not os.path.exists(video_path):
        logging.error(f"视频文件不存在: {video_path}")
        raise ValueError(f"视频文件不存在: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频: {video_path}")
        cap.release()
        raise ValueError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total-1, num_frames, dtype=int) if total >= num_frames else list(range(total)) + [total-1]*(num_frames-total)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(frames[-1] if frames else Image.new('RGB', (224,224)))
    cap.release()
    return frames

def preprocess_frames(keyframes: List[Image.Image]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return torch.stack([transform(f) for f in keyframes])


# ========================================
# 5. 推理核心
# ========================================
def load_model(model_path: str, agg_type: str = "attention", device=None) -> VideoPersonalityModel:
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = swin_config()
    swin = SwinTransformer(cfg, return_feat=True).to(device)
    model = VideoPersonalityModel(swin, agg_type=agg_type).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_single_video(model: VideoPersonalityModel, video_path: str, num_frames: int = 16, device=None) -> Tuple[float, str]:
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        frames = extract_keyframes(video_path, num_frames)
        frames_tensor = preprocess_frames(frames).unsqueeze(0).to(device)  # (1, T, C, H, W)
        with torch.no_grad():
            pred = model(frames_tensor)
        score = pred.squeeze().cpu().item()
        return score, "Extraversion"
    except Exception as e:
        logging.error(f"预测失败 {video_path}: {str(e)}")
        raise

def predict_batch_videos(model: VideoPersonalityModel, video_dir: str, num_frames: int = 16, device=None) -> pd.DataFrame:
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4']
    videos = [f for f in os.listdir(video_dir) if os.path.splitext(f)[1] in exts]
    results = []
    dim = "Extraversion"
    for v in videos:
        path = os.path.join(video_dir, v)
        try:
            score, _ = predict_single_video(model, path, num_frames, device)
            results.append({"video_name": v, dim: score})
            print(f"已处理: {v} -> {dim}: {score:.4f}")
        except Exception as e:
            print(f"失败: {v} -> {str(e)}")
    return pd.DataFrame(results)


# ========================================
# 6. 主函数
# ========================================
def main():
    config = {
        "model_path": "/root/autodl-tmp/swin-transformer/saved_models/Extraversion_Swin/model_epoch.pth",
        "agg_type": "attention",
        "num_frames": 16,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "single_video_path": None,
        "batch_video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Extraversion_test",
        "output_csv": "/root/autodl-tmp/swin-transformer/result/Extraversion_results.csv"
    }

    print(f"使用设备: {config['device']}")
    model = load_model(config["model_path"], config["agg_type"], config["device"])
    print(f"模型加载成功: {config['model_path']}")

    if config["single_video_path"] and os.path.exists(config["single_video_path"]):
        score, dim = predict_single_video(model, config["single_video_path"], config["num_frames"], config["device"])
        print(f"\n预测结果: {dim}: {score:.4f}")
    else:
        print(f"\n批量推理: {config['batch_video_dir']}")
        df = predict_batch_videos(model, config["batch_video_dir"], config["num_frames"], config["device"])
        os.makedirs(os.path.dirname(config["output_csv"]), exist_ok=True)
        df.to_csv(config["output_csv"], index=False)
        print(f"\n保存完成: {config['output_csv']}")
        print("前5条结果：")
        print(df.head())

if __name__ == "__main__":
    main()