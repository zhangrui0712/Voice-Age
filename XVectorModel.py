# -*- coding = utf-8 -*-
# @Time : 2024/6/15 17:55
# @Author : ZhangRui
# @File : XVectorModel.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import extract_mfcc

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

# 获取标签的数量
num_classes = len(os.listdir(DATASET_PATH))

# 初始化模型、损失函数和优化器
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
