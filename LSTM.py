import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import sklearn.preprocessing as preprocessing

dataset = pd.read_csv('champagne.csv')
dataset.columns = ['Month', 'Sales']
# 打印前5行数据
#print(dataset.head())

test_size = 24
train_set = dataset[:-test_size].Sales.values
test_set = dataset[-test_size:].Sales.values
#print(train_set)
#print(train_set.shape)
#print(test_set)
#print(test_set.shape)

#标准化
norm_scaler = preprocessing.StandardScaler()
train_set_normed = norm_scaler.fit_transform(train_set.reshape(-1,1)).reshape(-1, ).tolist()
#print(train_set_normed[:5])
#转tensor
train_set_normed = torch.FloatTensor(train_set_normed)
#print(train_set_normed[:5])

train_window_len = 28 # 训练窗口
pred_window_len = 4 #标签窗口
l = len(train_set_normed)
input_sequence = []
for i in range(l - train_window_len - pred_window_len + 1):
    train_seq = train_set_normed[i:i+train_window_len]
    label_seq = train_set_normed[i+train_window_len:i+train_window_len+pred_window_len]
    input_sequence.append((train_seq, label_seq))
print(len(input_sequence))
#第一个样本
print(input_sequence[0])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        # input_size表示输入的特征维数
        self.lstm = nn.LSTM(input_size, hidden_size)
        # output_size表示输出的特征维数
        self.linear = nn.Linear(hidden_size, output_size)
        # memory_cell
        self.hidden_cell = self.init_hidden_cell(output_size)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # 由于最终的预测值只保存在最后一个单元格中, 所以只要输出最后一个
        return self.linear(lstm_out.view(len(input_seq), -1))[-1]

    def init_hidden_cell(self, output_size):
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

model = LSTMModel(1, 100, pred_window_len)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 迭代次数
n_epochs = 30         #迭代次数经过调整，5 - 30 - 50
loss_line = []

for epoch in range(n_epochs):
    train_loss = 0.
    for seq, label in input_sequence:
        optimizer.zero_grad()

        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        model.hidden_cell = model.init_hidden_cell(pred_window_len)
        pred_label = model(seq)

        loss = criterion(pred_label, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'epoch{epoch + 1:2} loss: {train_loss / len(input_sequence):.4f}')
    loss_line.append(float(train_loss / len(input_sequence)))
#print(loss_line)

plt.figure()
plt.ylabel('loss')
plt.xlabel('iter_num')
plt.plot(range(n_epochs), loss_line,'b',label='loss')
plt.legend()
plt.savefig('image/loss.jpg')
#plt.show()

test_inputs = train_set_normed[-train_window_len:].tolist()
model.eval()
pred_value = [] # 用于存放预测值
with torch.no_grad():
    test_sequence = torch.FloatTensor(test_inputs)
    model.hidden_cell = model.init_hidden_cell(test_size)
    pred = model(test_sequence)
    #新版的sklearn需要数据为二位矩阵，需要reshape
    pred_value = pred.reshape(-1, 1)
# 由于前面对训练集进行了标准化，故预测结果都是小数，这里需要将预测结果反标准化
pred_value = norm_scaler.inverse_transform(pred_value).tolist()

plt.figure(figsize=(15,5),dpi = 500)
plt.grid(linestyle='--')
plt.plot(list(range(4)), test_set[:4], label='actual value')
plt.plot(list(range(4)), pred_value, label='predict value')
plt.legend()
plt.savefig('image/predict_fig.jpg')
#plt.show()

relative_error = 0.
for i in range(4):
    relative_error += (abs(pred_value[i] - test_set[i]) / test_set[i]) ** 2
acc = 1- np.sqrt(relative_error / 4)
print('Acc:', round(float(acc*100), 2), '%')