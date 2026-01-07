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
    filename="/root/autodl-tmp/DAN/saved_models/Extraversion/test_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------
# 1. 模型定义（与训练时一致）
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
# 2. 工具函数
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


def preprocess_frames(keyframes: List[Image.Image], img_size=(224, 224)) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    frame_tensors = [preprocess(frame) for frame in keyframes]
    return torch.stack(frame_tensors)


# --------------------------
# 3. 推理核心函数
# --------------------------
def load_model(model_path: str, agg_type: str = "attention", device: torch.device = None) -> VideoPersonalityModel:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建模型
    dan = DAN(
        features=make_layers(backbone['VGG16'], batch_norm=True),
        num_classes=1,
        return_feature=True
    )
    model = VideoPersonalityModel(dan, num_classes=1, agg_type=agg_type).to(device)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_single_video(model: VideoPersonalityModel, video_path: str, num_frames: int = 16, device: torch.device = None) -> Tuple[float, str]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        keyframes = extract_keyframes(video_path, num_frames)
        frames_tensor = preprocess_frames(keyframes)
        frames_tensor = frames_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(frames_tensor)
        
        score = outputs.squeeze(0).cpu().numpy().item()
        dimension = 'Extraversion'
        return score, dimension
    except Exception as e:
        logging.error(f"预测视频 {video_path} 失败: {str(e)}")
        raise


def predict_batch_videos(model: VideoPersonalityModel, video_dir: str, num_frames: int = 16, device: torch.device = None) -> pd.DataFrame:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(video_dir) if os.path.splitext(f)[1].lower() in video_extensions]
    
    results = []
    dimension = 'Extraversion'
    
    for video_name in video_files:
        video_path = os.path.join(video_dir, video_name)
        try:
            score, _ = predict_single_video(model, video_path, num_frames, device)
            result = {
                'video_name': video_name,
                dimension: score
            }
            results.append(result)
            print(f"已处理: {video_name} -> {dimension}: {score:.4f}")
        except Exception as e:
            print(f"处理失败 {video_name}: {str(e)}")
            continue
    
    return pd.DataFrame(results)


# --------------------------
# 4. 主推理流程
# --------------------------
def main():
    config = {
        "model_path": "/root/autodl-tmp/DAN/saved_models/Extraversion/model_epoch50.pth",
        "agg_type": "attention",
        "num_frames": 16,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "single_video_path": None,
        "batch_video_dir": "/root/autodl-tmp/data/opva/videos_mp4/Extraversion_test",
        "output_csv": "/root/autodl-tmp/DAN/result/Extraversion_tes_results.csv"
    }

    # 加载模型
    print(f"使用设备: {config['device']}")
    model = load_model(
        model_path=config["model_path"],
        agg_type=config["agg_type"],
        device=config["device"]
    )
    print(f"模型加载成功: {config['model_path']}")

    # 推理
    if config["single_video_path"] and os.path.exists(config["single_video_path"]):
        score, dimension = predict_single_video(
            model=model,
            video_path=config["single_video_path"],
            num_frames=config["num_frames"],
            device=config["device"]
        )
        print(f"\n单视频预测结果: {dimension}: {score:.4f}")

    else:
        print(f"\n开始批量处理文件夹: {config['batch_video_dir']}")
        results_df = predict_batch_videos(
            model=model,
            video_dir=config["batch_video_dir"],
            num_frames=config["num_frames"],
            device=config["device"]
        )
        
        os.makedirs(os.path.dirname(config["output_csv"]), exist_ok=True)
        results_df.to_csv(config["output_csv"], index=False)
        print(f"\n批量预测完成，结果保存至: {config['output_csv']}")
        print("前5条结果预览:")
        print(results_df.head())


if __name__ == "__main__":
    main()