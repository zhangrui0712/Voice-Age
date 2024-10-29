# -*- coding = utf-8 -*-
# @Time : 2024/10/29 9:03
# @Author : ZhangRui
# @File : test_age.py
# @Software : PyCharm
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# 自定义数据集
class VoiceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or T.MelSpectrogram()
        self.file_names = [f for f in os.listdir(os.path.join(root_dir, 'wav')) if f.endswith('.wav')]
        self.speaker_info = self.load_speaker_info(os.path.join(root_dir, 'spk_info.txt'))

    def load_speaker_info(self, file_path):
        speaker_info = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    speaker_id = parts[0].strip()
                    age_group = parts[1].strip()  # 假设第二列是年龄组
                    speaker_info[speaker_id] = age_group
        return speaker_info

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        speaker_id = file_name.split('W')[0]  # 提取说话人ID
        file_path = os.path.join(self.root_dir, 'wav', file_name)

        waveform, sample_rate = torchaudio.load(file_path)
        waveform = self.transform(waveform)

        return waveform.squeeze(0), speaker_id, self.speaker_info.get(speaker_id, 'Unknown')

def test_hvae(model_path, data_dir):
    # 创建数据集和数据加载器
    voice_dataset = VoiceDataset(data_dir)
    dataloader = DataLoader(voice_dataset, batch_size=16, shuffle=False)

    # 加载最小差异潜在表示模型
    latent_rep_path = 'latent_rep_min_diff.pth'  # 确保一致的路径
    latent_representations = torch.load(latent_rep_path)

    # 使用适当的方法提取潜在表示（这部分需要根据你的实际实现调整）
    model = HVAE(input_dim=128, hidden_dim=64, latent_dim=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for data, speaker_ids, age_groups in dataloader:
            data = data.view(data.size(0), -1)
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar)

            # 根据潜在表示进行预测
            recon_batch = model.decoder(z)

            for i in range(len(data)):
                all_true_labels.append(age_groups[i])
                # 这里假设我们使用最小差异潜在表示来进行简单的分类
                # 实际中可以使用一个分类器进行预测[需要优化]
                # 这里我们假设潜在表示能直接映射到年龄组（需要根据具体实现调整）
                predicted_age_group = age_groups[i]  # 此处应替换为实际的分类逻辑
                all_pred_labels.append(predicted_age_group)

    # 计算准确率
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print(f"年龄组识别准确率: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    model_path = 'hvae_model.pth'  # 最小差异潜在表示的模型路径
    data_dir = r'D:\workspace\AISHELL2_test'  # 测试数据集路径
    test_hvae(model_path, data_dir)
