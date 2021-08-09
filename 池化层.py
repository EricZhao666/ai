# 池化层的正向传播
import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 最大
pool2d(X, (2, 2))
'''
tensor([[4., 5.],
        [7., 8.]])
'''
# 平均
pool2d(X, (2, 2), 'avg')
'''
tensor([[2., 3.],
        [5., 6.]])
'''
# 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
'''
# 步幅形状为 (3, 3)
pool2d = nn.MaxPool2d(3)
pool2d(X)
'''
tensor([[[[10.]]]])
'''
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
'''
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
'''
pool2d = nn.MaxPool2d((2, 3), padding=(1, 1), stride=(2, 3))
pool2d(X)
'''
tensor([[[[ 1.,  3.],
          [ 9., 11.],
          [13., 15.]]]])
'''
# 多通道
X = torch.cat((X, X + 1), 1)
'''
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])
'''
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
'''
同样是两个输出
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
'''