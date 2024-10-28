# -*- coding = utf-8 -*-
# @Time : 2024/6/15 17:52
# @Author : ZhangRui
# @File : extract_mfcc.py
# @Software : PyCharm
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 定义音频文件路径
DATASET_PATH = "LibriSpeech/train-clean-5"

# 提取MFCC特征
def extract_mfcc(audio_path, n_mfcc=40, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

class LibriSpeechDataset(Dataset):
    def __init__(self, dataset_path):
        self.file_list = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".flac"):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mfcc = extract_mfcc(self.file_list[idx])
        label = int(self.file_list[idx].split('/')[-2])  # 假设目录名是标签
        return torch.tensor(mfcc, dtype=torch.float32), label

# 创建数据集实例
dataset = LibriSpeechDataset(DATASET_PATH)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
print("========loading datasets ...==========")
print("train_size = ",train_size)
print("test_size = ",test_size)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
