# -*- coding = utf-8 -*-
# @Time : 2024/6/15 17:55
# @Author : ZhangRui
# @File : test.py
# @Software : PyCharm
# 加载模型

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
