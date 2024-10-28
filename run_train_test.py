# -*- coding = utf-8 -*-
# @Time : 2024/6/16 21:07
# @Author : ZhangRui
# @File : run_train_test.py
# @Software : PyCharm
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 定义音频文件路径
DATASET_PATH = "LibriSpeech/train-clean-5"
MAX_MFCC_LENGTH = 500  # 定义最大MFCC长度，根据你的数据集调整

# 提取MFCC特征
def extract_mfcc(audio_path, n_mfcc=40, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

class LibriSpeechDataset(Dataset):
    def __init__(self, dataset_path):
        self.file_list = []
        self.labels = []
        self.label_map = {}
        label_counter = 0
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".flac"):
                    self.file_list.append(os.path.join(root, file))
                    label_str = os.path.basename(root)  # 获取父目录名作为标签
                    if label_str not in self.label_map:
                        self.label_map[label_str] = label_counter
                        label_counter += 1
                    self.labels.append(self.label_map[label_str])
        self.num_classes = len(self.label_map)
        print(f"Number of classes: {self.num_classes}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mfcc = extract_mfcc(self.file_list[idx])
        # 填充或截断MFCC特征
        if (mfcc.shape[0] > MAX_MFCC_LENGTH):
            mfcc = mfcc[:MAX_MFCC_LENGTH, :]
        else:
            padding = np.zeros((MAX_MFCC_LENGTH - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack((mfcc, padding))
        label = self.labels[idx]
        return torch.tensor(mfcc, dtype=torch.float32), label

# 自定义collate_fn，用于数据加载器
def collate_fn(batch):
    mfccs, labels = zip(*batch)
    mfccs = torch.stack(mfccs)
    labels = torch.tensor(labels)
    return mfccs, labels

print("========loading datasets ...==========")
# 创建数据集实例
dataset = LibriSpeechDataset(DATASET_PATH)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 定义x-vector模型
import torch
import torch.nn as nn
import torch.optim as optim

class XVectorModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(XVectorModel, self).__init__()
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=7, stride=1, padding=3)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.tdnn1(x)
        x = self.relu(x)
        x = self.tdnn2(x)
        x = self.relu(x)
        x = self.tdnn3(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)  # Temporal pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

print("========training datasets ...==========")
# 初始化模型、损失函数和优化器
num_classes = dataset.num_classes
model = XVectorModel(input_dim=40, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for mfccs, labels in train_loader:
        mfccs = mfccs.permute(0, 2, 1)  # (batch_size, time, features) -> (batch_size, features, time)
        optimizer.zero_grad()
        outputs = model(mfccs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# 保存模型
torch.save(model.state_dict(), 'xvector_model.pth')

# 评估和测试模型
model.load_state_dict(torch.load('xvector_model.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for mfccs, labels in test_loader:
        mfccs = mfccs.permute(0, 2, 1)
        outputs = model(mfccs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
