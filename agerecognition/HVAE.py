# -*- coding = utf-8 -*-
# @Time : 2024/10/28 16:56
# @Author : ZhangRui
# @File : HVAE.py
# @Software : PyCharm
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# 自定义数据集
class VoiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.MelSpectrogram()
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.root_dir, file_name)

        # 加载音频文件
        waveform, sample_rate = torchaudio.load(file_path)

        # 进行必要的变换（如特征提取）
        waveform = self.transform(waveform)

        return waveform.squeeze(0), file_name  # 返回波形和文件名


# HVAE 模型定义
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h2 = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h2))


class HVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(HVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# 设置超参数
input_dim = 128  # 根据特征提取的输出维度调整
hidden_dim = 64
latent_dim = 32
epochs = 10
batch_size = 16
learning_rate = 1e-3

# 创建数据集和数据加载器
root_dir = r'LibriSpeech/train-clean-5'
voice_dataset = VoiceDataset(root_dir)
dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器
model = HVAE(input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
model.train()
for epoch in range(epochs):
    train_loss = 0
    for data, _ in dataloader:
        data = data.view(data.size(0), -1)  # 展平输入
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')

# 保存训练好的模型
torch.save(model.state_dict(), 'hvae_model.pth')
print("模型已保存！")
