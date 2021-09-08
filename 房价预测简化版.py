import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
'''
该版本为直接运用sklearn.linear_model里的LinearRegression计算得到结果
'''
# 读入提前下好的数据，来自kaggle
train_data = pd.read_csv('kaggle_house_pred_train.csv')
test_data = pd.read_csv('kaggle_house_pred_test.csv')
'''
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
'''
# 删除id，将所有的训练和测试数据连在一起
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)
# 处理缺失的数据
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有数据都意味着消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 将类别变量转换为伪变量
all_features = pd.get_dummies(all_features, dummy_na=True)
# 转换为张量方便训练
n_train = train_data.shape[0]  # 得到共有多少行
# train_feature_1 = all_features[:n_train]
# print(train_feature_1.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
train_features=all_features[:n_train]
train_labels=train_data.iloc[:,[-1]]
test_features=all_features[n_train:]
from sklearn.model_selection import train_test_split
from ml_metrics import rmse
# 将训练数据拆分进行验证
x_train, x_validation, y_train, y_validation = train_test_split(train_features ,train_labels,  train_size=0.8, test_size=0.2, random_state =20)
#Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(train_features)
pred_test=model.predict(test_features)
print(pred)
print(train_labels)
print(pred_test)
'''
#Prediction
y_train_predicted = model.predict(x_train)  # RMSE on train
y_valid_predicted = model.predict(x_validation)  # RMSE on validation
print("The RMSE on train is : ",rmse(y_train,y_train_predicted))
print("The RMSE on Validation is : ",rmse(y_validation,y_valid_predicted))
'''


