import torch
'''
x=torch.arange(12)# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
x.shape# 可以通过张量的 shape 属性来访问张量的形状 （沿每个轴的长度）torch.Size([12])
x.numel()# 12 x里的总数
X = x.reshape(3, 4)
print(X)
'''
'''
torch.zeros((2, 3, 4))# 创建一个形状为(2,3,4)的张量 内容全为零
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])#创建张量
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y  # **运算符是求幂运算
'''
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)#cat表示将两个张量合并，dim决定哪个纬度合并
'''
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [ 2.,  1.,  4.,  3.],
         [ 1.,  2.,  3.,  4.],
         [ 4.,  3.,  2.,  1.]]),
 tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
'''
#广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
'''
(tensor([[0],
         [1],
         [2]]),
 tensor([[0, 1]]))
'''
a + b
'''
由于 a 和 b 分别是  3×1  和  1×2  矩阵，如果我们让它们相加，它们的形状不匹配。
我们将两个矩阵广播为一个更大的  3×2  矩阵，如下所示：矩阵 a将复制列，矩阵 b将复制行，然后再按元素相加。
tensor([[0, 1],
        [1, 2],
        [2, 3]])
'''
#reshape后用的是同一个内存地址
'''
a=torch.arange(12)
b=a.reshape((3,4))
b[:]=2
a
此时a也会改变
'''