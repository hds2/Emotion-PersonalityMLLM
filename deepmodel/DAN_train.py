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
import logging

# 设置日志
logging.basicConfig(
    filename="/root/autodl-tmp/DAN/saved_models/Conscientiousness/training_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------
# 1. 模型定义
# --------------------------
backbone = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

class DAN(nn.Module):
    def __init__(self, features, num_classes=1, init_weights=True, return_feature=False):
        super(DAN, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.maxpool = nn.AdaptiveMaxPool2d((7, 7))
        self.linear_1 = nn.Linear(50176, 2048)
        self.linear_mid = nn.Linear(2048, 1024)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(1024, num_classes)
        self.return_feature = return_feature
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x1 = self.avgpool(x)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = self.maxpool(x)
        x2 = F.normalize(x2, p=2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.leaky_relu(x)
        feat = self.linear_mid(x)
        x = self.leaky_relu(feat)
        x = self.dropout(x)
        x = self.linear_2(x)
        if self.return_feature:
            return x, feat
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VideoPersonalityModel(nn.Module):
    def __init__(self, dan_model: nn.Module, num_classes=1, agg_type="attention"):
        super(VideoPersonalityModel, self).__init__()
        self.dan = dan_model
        self.num_classes = num_classes
        self.agg_type = agg_type
        if self.agg_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, 1)
            )
        elif self.agg_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.fc_lstm = nn.Linear(1024, 1024)

    def forward(self, video_frames: torch.Tensor):
        B, T, C, H, W = video_frames.shape
        frame_outputs, frame_features = self.dan(video_frames.view(B*T, C, H, W))
        frame_features = frame_features.view(B, T, 1024)
        if self.agg_type == "mean":
            video_feat = frame_features.mean(dim=1)
        elif self.agg_type == "attention":
            att_weights = self.attention(frame_features)
            att_weights = F.softmax(att_weights, dim=1)
            video_feat = torch.sum(frame_features * att_weights, dim=1)
        elif self.agg_type == "lstm":
            lstm_out, _ = self.lstm(frame_features)
            video_feat = self.fc_lstm(lstm_out[:, -1, :])
        video_feat = self.dan.leaky_relu(video_feat)
        video_feat = self.dan.dropout(video_feat)
        video_pred = self.dan.linear_2(video_feat)
        return video_pred


# --------------------------
# 2. 视频处理工具函数
# --------------------------
def extract_keyframes(video_path: str, num_frames: int = 16) -> List[Image.Image]:
    if not os.path.exists(video_path):
        logging.error(f"视频文件不存在: {video_path}")
        raise ValueError(f"视频文件不存在: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"无法打开视频文件: {video_path}")
        cap.release()
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
        while len(frame_indices) < num_frames:
            frame_indices.append(total_frames - 1)
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    keyframes = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keyframes.append(Image.fromarray(frame_rgb))
        else:
            if keyframes:
                keyframes.append(keyframes[-1])
            else:
                logging.error(f"无法读取视频帧: {video_path}")
                cap.release()
                raise ValueError(f"无法读取视频帧: {video_path}")
    cap.release()
    return keyframes


def get_train_transform(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# --------------------------
# 3. 数据集
# --------------------------
class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_dir: str, num_frames: int = 16, is_train=True):
        self.df = df
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.is_train = is_train
        self.train_transform = get_train_transform()
        self.valid_indices = self._validate_videos()

    def _validate_videos(self):
        valid_indices = []
        for idx in range(len(self.df)):
            video_name = self.df.iloc[idx]['video_name']
            video_path = os.path.join(self.video_dir, video_name)
            try:
                if not os.path.exists(video_path):
                    logging.error(f"视频文件不存在: {video_path}")
                    continue
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logging.error(f"无法打开视频文件: {video_path}")
                    cap.release()
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    logging.error(f"视频文件为空: {video_path}")
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, _ = cap.read()
                if not ret:
                    logging.error(f"无法读取视频帧: {video_path}")
                    cap.release()
                    continue
                valid_indices.append(idx)
                cap.release()
            except Exception as e:
                logging.error(f"验证视频文件 {video_path} 时出错: {str(e)}")
                continue
        print(f"有效视频文件数: {len(valid_indices)}/{len(self.df)}")
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
        video_name = row['video_name']
        video_path = os.path.join(self.video_dir, video_name)
        
        try:
            keyframes = extract_keyframes(video_path, self.num_frames)
        except ValueError as e:
            logging.error(f"无法加载视频 {video_path}: {str(e)}")
            frames_tensor = torch.zeros((self.num_frames, 3, 224, 224))
            label = torch.tensor(row['Conscientiousness'], dtype=torch.float32).unsqueeze(0)
            return frames_tensor, label

        frame_tensors = [self.train_transform(frame) for frame in keyframes]
        frames_tensor = torch.stack(frame_tensors)
        
        label = torch.tensor(row['Conscientiousness'], dtype=torch.float32).unsqueeze(0)
        
        return frames_tensor, label


# --------------------------
# 4. 训练函数
# --------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    for batch in tqdm(dataloader, desc="Training"):
        frames, labels = batch
        frames = frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        
        mae = torch.mean(torch.abs(outputs - labels))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * frames.size(0)
        total_mae += mae.item() * frames.size(0)
        total_samples += frames.size(0)
        
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
    return avg_loss, avg_mae


# --------------------------
# 5. 主训练流程
# --------------------------
def main():
    config = {
        "video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Conscientiousness_train",
        "data_csv": "/root/autodl-tmp/data/conscientiousness_train.csv",
        "num_frames": 16,
        "agg_type": "attention",
        "pretrained_dan": True,
        "batch_size": 8,
        "epochs": 50,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "save_dir": "/root/autodl-tmp/DAN/saved_models/Conscientiousness",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    os.makedirs(config["save_dir"], exist_ok=True)

    # 加载数据集
    full_df = pd.read_csv(config["data_csv"])
    required_columns = ['video_name', 'Conscientiousness']
    if not set(required_columns).issubset(full_df.columns):
        missing = set(required_columns) - set(full_df.columns)
        raise ValueError(f"标签文件缺少必要的列: {missing}")
    
    # 所有数据用于训练，无验证集
    train_df = full_df

    # 构建数据集和加载器
    train_dataset = VideoDataset(
        train_df, 
        config["video_dir"], 
        config["num_frames"], 
        is_train=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 构建模型
    print(f"使用设备: {config['device']}")
    dan = DAN(
        features=make_layers(backbone['VGG16'], batch_norm=True),
        num_classes=1,
        return_feature=True
    )
    if config["pretrained_dan"]:
        try:
            pretrained_dict = torch.load("./pre_trained_weights/vgg16_bn-6c64b313.pth", map_location='cpu')
            model_dict = dan.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            dan.load_state_dict(model_dict)
            for param in list(dan.features.parameters())[:10]:
                param.requires_grad = False
            print("成功加载VGG预训练权重并冻结底层")
        except FileNotFoundError:
            print("未找到预训练权重，使用随机初始化")

    model = VideoPersonalityModel(
        dan, 
        num_classes=1,
        agg_type=config["agg_type"]
    ).to(config["device"])

    # 训练配置
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # 训练循环
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss, train_mae = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            config["device"]
        )
        
        print(f"训练损失: {train_loss:.4f} | 训练MAE: {train_mae:.4f}")
        
        # 保存每个epoch的模型
        save_path = os.path.join(config["save_dir"], f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"保存模型至: {save_path}")

        scheduler.step()

    print("训练完成！")


if __name__ == "__main__":
    main()