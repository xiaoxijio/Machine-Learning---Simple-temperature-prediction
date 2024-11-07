import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import datetime

from matplotlib import pyplot as plt
from sklearn import preprocessing

features = pd.read_csv('data/temps.csv')  # 导入数据
# print(features.head())

features = pd.get_dummies(features)  # 独热编码 例如疯狂星期四为 [0,0,0,1,0,0,0]
# print(features.head())

labels = np.array(features['actual'])  # 标签
features = features.drop('actual', axis=1)  # 从特征中删去标签
feature_list = list(features.columns)
features = np.array(features)

input_features = preprocessing.StandardScaler().fit_transform(features)  # 标准化, 去量纲
# print(input_features[0])

# 练习版本 帮助理解过程
# x = torch.tensor(input_features, dtype=torch.float64)  # 输入特征
# y = torch.tensor(labels, dtype=torch.float64)  # 输出标签
# w1 = torch.randn((14, 128), dtype=torch.float64, requires_grad=True)  # 数据大小[348,14]*权重[14,128]+偏置[128]
# b1 = torch.randn(128, dtype=torch.float64, requires_grad=True)
# w2 = torch.randn((128, 1), dtype=torch.float64, requires_grad=True)
# b2 = torch.randn(1, dtype=torch.float64, requires_grad=True)
#
# lr = 0.001
#
# for i in range(1000):
#     hidden = x.mm(w1) + b1  # y = wx + b [348,128]
#     hidden = torch.relu_(hidden)  # 激活！
#     predictions = hidden.mm(w2) + b2  # y = wx + b [348, 1]
#     loss = torch.mean((predictions - y) ** 2)  # 损失 = (预测值 - 真实值) ^ 2
#
#     if i % 100 == 0:
#         print('loss:', loss)
#     loss.backward()  # 反向传播
#
#     w1.data.add_(-lr * w1.grad.data)  # 等价 w1 = w1 - lr * w1.grad
#     b1.data.add_(-lr * b1.grad.data)  # w1 = w1 − lr × ∇L(w1)
#     w2.data.add_(-lr * w2.grad.data)
#     b2.data.add_(-lr * b2.grad.data)
#
#     w1.grad.data.zero_()  # 迭代清空
#     b1.grad.data.zero_()
#     w2.grad.data.zero_()
#     b2.grad.data.zero_()


input_size = input_features.shape[1]  # 输入层
hidden_size = 128  # 隐藏层
output_size = 1  # 输出层
batch_size = 16  # 批次大小
x = torch.tensor(input_features, dtype=torch.float32)  # 输入特征
y = torch.tensor(labels, dtype=torch.float32)  # 输出标签

my_nn = torch.nn.Sequential(  # 自定义的训练模型
    torch.nn.Linear(input_size, hidden_size),  # 全连接层 也称线性层
    torch.nn.Sigmoid(),  # 激活
    torch.nn.Linear(hidden_size, output_size)
)
cost = torch.nn.MSELoss(reduction='mean')  # 损失函数
optimizer = optim.Adam(my_nn.parameters(), lr=0.001)  # 优化器

for i in range(1000):
    for start in range(0, len(input_features), batch_size):  # 小批次训练
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = x[start:end]
        yy = y[start:end]
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    if i % 100 == 0:
        print('loss:', loss)

predict = my_nn(x).data.numpy()  # 对比训练出的数据与原数据
# 画图 日期为横坐标 原数据标签与预测值为纵坐标
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
predictions_data = pd.DataFrame(data={'date': dates, 'prediction': predict.reshape(-1)})

plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
