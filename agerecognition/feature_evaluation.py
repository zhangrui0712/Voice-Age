# -*- coding = utf-8 -*-
# @Time : 2024/10/29 8:38
# @Author : ZhangRui
# @File : feature_evaluation.py
# @Software : PyCharm
import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.io.wavfile import write


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


def save_audio(waveform, filename, sample_rate=16000):
    """保存还原的音频文件"""
    write(filename, sample_rate, waveform.cpu().numpy())


def compute_difference(original, reconstructed):
    """计算原始音频与还原音频之间的差异"""
    return F.mse_loss(reconstructed, original).item()


# 主评估函数
def evaluate_hvae(model_path, data_dir):
    # 创建数据集和数据加载器
    voice_dataset = VoiceDataset(data_dir)
    dataloader = DataLoader(voice_dataset, batch_size=16, shuffle=True)

    # 加载模型
    model = HVAE(input_dim=128, hidden_dim=64, latent_dim=32)  # 确保与训练时一致
    model.load_state_dict(torch.load(model_path))
    model.eval()

    differences = {}
    selected_latent_representations = {}

    with torch.no_grad():
        for data, speaker_ids, age_groups in dataloader:
            data = data.view(data.size(0), -1)
            mu, logvar = model.encoder(data)
            z = model.reparameterize(mu, logvar)

            # 还原语音
            recon_batch = model.decoder(z)

            for i in range(len(data)):
                original_waveform = data[i].view(1, -1)
                reconstructed_waveform = recon_batch[i].view(1, -1)

                save_audio(reconstructed_waveform, f'reconstructed_{speaker_ids[i]}.wav')
                difference = compute_difference(original_waveform, reconstructed_waveform)

                age_group = age_groups[i]
                if age_group not in differences:
                    differences[age_group] = []
                differences[age_group].append((speaker_ids[i], difference, z[i]))

                # 选择最小差异的潜在表示
                if age_group not in selected_latent_representations or difference < \
                        selected_latent_representations[age_group][0]:
                    selected_latent_representations[age_group] = (difference, z[i])

    # 将差异记录到文件
    with open('differences_by_age_group.txt', 'w') as f:
        for age_group, files in differences.items():
            f.write(f"Age Group: {age_group}\n")
            for filename, diff, _ in files:
                f.write(f"  {filename}: {diff}\n")

    # 保存最小差异潜在表示到第二个模型文件
    for age_group, (min_diff, latent_rep) in selected_latent_representations.items():
        latent_rep_filename = f'selected_latent_rep_{age_group}.pth'
        torch.save(latent_rep, latent_rep_filename)
        print(f"最小差异潜在表示已保存为 '{latent_rep_filename}'，年龄组: {age_group}")


if __name__ == '__main__':
    model_path = 'hvae_model.pth'  # 训练好的模型路径
    data_dir = r'D:\workspace\AISHELL2'  # 数据集路径
    evaluate_hvae(model_path, data_dir)
