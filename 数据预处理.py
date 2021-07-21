import os
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd
data = pd.read_csv(data_file)
print(data)
#接下来数据预处理
'''
NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN 第一步处理结果
'''
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())#将房间数取均值填入
print(inputs)
'''
对于 inputs 中的类别值或离散值，我们将 “NaN” 视为一个类别。
由于 “巷子”（“Alley”）列只接受两种类型的类别值 “Pave” 和 “NaN”，pandas 可以自动将此列转换为两列 “Alley_Pave” 和 “Alley_nan”。
巷子类型为 “Pave” 的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。
缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

'''
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
'''
NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
'''
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
'''
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
'''
