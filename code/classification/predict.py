import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset import testDataset
from LSTM import stockLSTM
import os
input_size = 16
hidden_size = 128
num_layers = 3
output_size = 6
ts_set=testDataset(os.path.join("dataset","test"))
test_data = DataLoader(dataset=ts_set,batch_size=1)
model = stockLSTM(input_size, hidden_size, num_layers, output_size)
model.eval()
model.load_state_dict((torch.load("checkpoint/lstm39.pth")))
# 函数用于将模型对数据集的预测结果保存为CSV文件
def save_predictions_to_csv(model, dataset, output_csv='predictions.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)
    predictions = []
    file_names = []

    for i, data in enumerate(dataset):

        file_name, input_data = data  # 文件名和数据
        if input_data.size()[1]==0:
            print("empty")
            continue
        input_data = input_data.to(device)
        output = model(input_data)  # 增加一个维度进行预测

        # 使用softmax获取概率分布
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()

        file_names.append(file_name)
        predictions.append(predicted_class)

    # 将文件名和预测结果合并为 DataFrame
    data = {'File Name': file_names, 'Prediction': predictions}
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to '{output_csv}'")



def predict_and_save(model, dataset, output_csv='predictions.csv'):
    save_predictions_to_csv(model, dataset, output_csv)
if __name__ == "__main__":
    predict_and_save(model,test_data)
