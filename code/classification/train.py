import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import preprocess
from LSTM import stockLSTM
from dataset import CustomDataset
import os
# 在这里定义你的 CustomDataset 和 stockLSTM 模型类...

# 定义你的数据集和模型
root_directory = os.path.join("","dataset","top")
dataset = CustomDataset(root_directory)

input_size = 16
hidden_size = 128
num_layers = 3
output_size = 6

model = stockLSTM(input_size, hidden_size, num_layers, output_size)

# 定义超参数和训练配置
learning_rate = 0.001
batch_size = 1
num_epochs = 40

# 数据加载器
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 检查是否可用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("device",device)

# 训练循环
for epoch in range(num_epochs):
    total_loss = 0
    for i, (data, target) in enumerate(dataloader):
        if data.size()[1]==0:
            print("empty")
            continue
        data = data.to(device)
        target = target.to(device)

        # 前向传播
        print(data.size())
        outputs = model(data)
        loss = criterion(outputs, target)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {total_loss / 100}')
        total_loss=0
        # 每100步打印损失

    torch.save(model.state_dict(), f'checkpoint/lstm{epoch}.pth')

print('训练结束！')
