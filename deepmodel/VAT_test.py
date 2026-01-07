# test.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import math
import logging

# ========================================
# 1. 日志
# ========================================
os.makedirs("/root/autodl-tmp/VAT/savel_model/Extraversion", exist_ok=True)
logging.basicConfig(
    filename="/root/autodl-tmp/VAT/savel_model/Extraversion/test_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

# ========================================
# 2. 必须与训练一致模型结构
# ========================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, dropout=0.3):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        q = q.unsqueeze(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out.squeeze(1)

class TX(nn.Module):
    def __init__(self, d_k, dropout=0.3):
        super().__init__()
        self.attn = MultiHeadAttention(d_k, dropout)
        self.norm1 = nn.LayerNorm(d_k)
        self.norm2 = nn.LayerNorm(d_k)
        self.ff = FeedForward(d_k, d_ff=4*d_k, dropout=dropout)
    def forward(self, q, k, v, mask=None):
        q = self.norm1(q + self.attn(q, k, v, mask))
        q = self.norm2(q + self.ff(q))
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
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        nn.init.normal_(self.classifier.weight, std=0.001)
    def forward(self, x, b, t):
        x = self.bn(x)
        x = self.proj(x).squeeze(-1).squeeze(-1)
        x = x.view(b, t, self.d_model)
        x = self.pos_enc(x)
        q = x.mean(dim=1)
        k = v = x
        q = q.view(b, self.head, self.d_k)
        k = k.view(b, t, self.head, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.head, self.d_k).transpose(1, 2)
        head_outs = [self.heads[i](q[:, i], k[:, i], v[:, i]) for i in range(self.head)]
        f = torch.cat(head_outs, dim=-1)
        f = F.normalize(f, p=2, dim=1)
        y = self.classifier(f)
        y = torch.sigmoid(y) * 4.0 + 1.0
        return (y, f) if self.return_feature else y

class SemiTransformer(nn.Module):
    def __init__(self, num_classes=1, seq_len=16, return_feature=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.tail = Tail(num_classes=num_classes, num_frames=seq_len, return_feature=return_feature)
        for p in list(self.base.parameters())[:5]:
            p.requires_grad = False
    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, *x.shape[2:])
        x = self.base(x)
        return self.tail(x, b, t)

# ========================================
# 3. 推理工具
# ========================================
def extract_keyframes(video_path: str, num_frames: int = 16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: raise ValueError("Empty video")
    indices = np.linspace(0, total-1, num_frames, dtype=int) if total >= num_frames else list(range(total)) + [total-1]*(num_frames-total)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        else:
            frames.append(frames[-1])
    cap.release()
    return frames

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ========================================
# 4. 推理函数
# ========================================
def load_model(model_path: str, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemiTransformer(num_classes=1, seq_len=16).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"模型加载成功: {model_path}")
    return model

def predict_single_video(model, video_path: str, num_frames=16, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = extract_keyframes(video_path, num_frames)
    transform = get_transform()
    tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return round(pred.item(), 4)

def predict_batch(model, video_dir: str, num_frames=16, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    results = []
    for v in videos:
        path = os.path.join(video_dir, v)
        try:
            score = predict_single_video(model, path, num_frames, device)
            results.append({"video_name": v, "Conscientiousness": score})
            print(f"{v} -> {score}")
        except Exception as e:
            print(f"Failed {v}: {e}")
    return pd.DataFrame(results)

# ========================================
# 5. 主函数
# ========================================
def main():
    config = {
        "model_path": "/root/autodl-tmp/VAT/savel_model/Extraversion/epoch50.pth",
        "video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Extraversion_test",
        "output_csv": "/root/autodl-tmp/VAT/result/Extraversion_predictions.csv",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    model = load_model(config["model_path"], config["device"])
    df = predict_batch(model, config["video_dir"], num_frames=16, device=config["device"])
    os.makedirs(os.path.dirname(config["output_csv"]), exist_ok=True)
    df.to_csv(config["output_csv"], index=False)
    print(f"\n预测完成！结果保存至: {config['output_csv']}")
    print(df.head())

if __name__ == "__main__":
    main()