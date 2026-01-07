# train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import csv
from typing import List   


# ========================================
# 1. 模型定义（标准 Multi-Head Attention + LayerNorm）
# ========================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, dropout=0.3):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        q = q.unsqueeze(1)  # (b, 1, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (b,1,t)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (b,1,d_k)
        return out.squeeze(1)  # (b, d_k)

class TX(nn.Module):
    def __init__(self, d_k, dropout=0.3):
        super().__init__()
        self.attn = MultiHeadAttention(d_k, dropout)
        self.norm1 = nn.LayerNorm(d_k)
        self.norm2 = nn.LayerNorm(d_k)
        self.ff = FeedForward(d_k, d_ff=4*d_k, dropout=dropout)
    def forward(self, q, k, v, mask=None):
        attn_out = self.attn(q, k, v, mask)
        q = self.norm1(q + attn_out)
        ff_out = self.ff(q)
        q = self.norm2(q + ff_out)
        return q

class BlockHead(nn.Module):
    def __init__(self, d_k, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([TX(d_k, dropout) for _ in range(3)])
    def forward(self, q, k, v, mask=None):
        for layer in self.layers:
            q = layer(q, k, v, mask)
        return q

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=80):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class Tail(nn.Module):
    def __init__(self, num_classes=1, num_frames=16, head=16, return_feature=False):
        super().__init__()
        self.num_frames = num_frames
        self.head = head
        self.d_model = 1024
        self.d_k = self.d_model // head
        self.return_feature = return_feature

        self.bn = nn.BatchNorm2d(2048)
        self.proj = nn.Conv2d(2048, self.d_model, kernel_size=(7, 4), bias=False)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=num_frames + 1)
        self.heads = nn.ModuleList([BlockHead(d_k=self.d_k, dropout=0.3) for _ in range(head)])
        self.classifier = nn.Linear(self.d_model, num_classes)

        # 初始化
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x, b, t):
        x = self.bn(x)
        x = self.proj(x).squeeze(-1).squeeze(-1)
        x = x.view(b, t, self.d_model)
        x = self.pos_enc(x)

        # 关键：中间帧作为 query
        q = x[:, t // 2, :]  # (b, 1024)
        k = v = x

        q = q.view(b, self.head, self.d_k)
        k = k.view(b, t, self.head, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.head, self.d_k).transpose(1, 2)

        head_outs = [self.heads[i](q[:, i], k[:, i], v[:, i]) for i in range(self.head)]
        f = torch.cat(head_outs, dim=-1)
        # 去掉 normalize
        y = self.classifier(f)
        y = torch.sigmoid(y) * 4.0 + 1.0
        return y

class SemiTransformer(nn.Module):
    def __init__(self, num_classes=1, seq_len=16, return_feature=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.tail = Tail(num_classes=num_classes, num_frames=seq_len, return_feature=return_feature)
        # 冻结前5层
        for p in list(self.base.parameters())[:5]:
            p.requires_grad = False
    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, *x.shape[2:])
        x = self.base(x)
        return self.tail(x, b, t)

# ========================================
# 2. 数据处理
# ========================================
def extract_keyframes(video_path: str, num_frames: int = 16) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError(f"Cannot open video: {video_path}")
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(frames[-1])
    cap.release()
    return frames

def get_transform(is_train=True):
    base = [
        transforms.Resize((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if is_train:
        base.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(base)

class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str, num_frames: int = 16, is_train=True):
        self.df = df.reset_index(drop=True)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = get_transform(is_train)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.video_dir, row['video_name'])
        frames = extract_keyframes(video_path, self.num_frames)
        tensor = torch.stack([self.transform(f) for f in frames])
        label = (row['Extraversion'] - 1.0) / 4.0
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return tensor, label

# ========================================
# 3. 训练循环
# ========================================
def train_one_epoch(model, loader, criterion_mse, criterion_l1, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = total_mae = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(frames)  # [0,1]
        loss_mse = criterion_mse(preds, labels)
        loss_l1 = criterion_l1(preds, labels)
        loss = loss_mse + 0.5 * loss_l1
        loss.backward()
        optimizer.step()
        batch_size = frames.size(0)
        total_loss += loss.item() * batch_size
        total_mae += torch.abs(preds - labels).mean().item() * batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{torch.abs(preds-labels).mean().item():.4f}'})
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

# ========================================
# 4. 主函数
# ========================================
def main():
    C = {
        "video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Extraversion_train",
        "data_csv": "/root/autodl-tmp/data/extraversion_train.csv",
        "num_frames": 16,
        "batch_size": 4,
        "epochs": 50,
        "save_dir": "/root/autodl-tmp/VAT/savel_model/Extraversion",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    os.makedirs(C["save_dir"], exist_ok=True)

    df = pd.read_csv(C["data_csv"])
    if not {'video_name', 'Extraversion'}.issubset(df.columns):
        raise ValueError("CSV 必须包含 'video_name' 和 'Extraversion'")

    print("\n标签统计:")
    print(df['Extraversion'].describe())

    dataset = VideoDataset(df, C["video_dir"], C["num_frames"], is_train=True)
    loader = DataLoader(dataset, batch_size=C["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    model = SemiTransformer(num_classes=1, seq_len=C["num_frames"]).to(C["device"])

    base_params = [p for n, p in model.base.named_parameters() if p.requires_grad]
    tail_params = list(model.tail.parameters())
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': 1e-3},
        {'params': tail_params, 'lr': 1e-4}
    ], weight_decay=1e-5)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    log_path = os.path.join(C["save_dir"], "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_mae"])

    best_mae = float('inf')
    for epoch in range(1, C["epochs"] + 1):
        loss, mae = train_one_epoch(model, loader, criterion_mse, criterion_l1, optimizer, C["device"], epoch, C["epochs"])
        scheduler.step()
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{loss:.5f}", f"{mae:.5f}"])
        print(f"\nEpoch {epoch}/{C['epochs']} | Loss: {loss:.5f} | MAE: {mae:.5f}")

        torch.save(model.state_dict(), os.path.join(C["save_dir"], f"epoch{epoch}.pth"))
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), os.path.join(C["save_dir"], "best_model.pth"))

    print(f"\n训练完成！最佳模型: {C['save_dir']}/best_model.pth")
    print(f"日志: {log_path}")

if __name__ == "__main__":
    main()