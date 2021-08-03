import torch
from torch import nn
from torch.nn import functional as F
# 多层感知机的直接利用库实现
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
net(X)
'''
tensor([[-0.1220, -0.1627, -0.0651,  0.0356,  0.1121,  0.0898,  0.1654,  0.1972,
         -0.0966,  0.0048],
        [-0.0544, -0.1490, -0.1204,  0.0516,  0.2844,  0.1056,  0.0416,  0.1751,
          0.0045, -0.0097]], grad_fn=<AddmmBackward>)
'''
# 自定义块
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Block`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的正向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
'''
net = MLP()
net(X)
tensor([[-0.2222,  0.0756, -0.1518,  0.2255,  0.0368, -0.1240, -0.0133,  0.0106,
          0.2335,  0.0833],
        [-0.0614,  0.0977, -0.0852,  0.2003, -0.0314, -0.1109,  0.0949,  0.0333,
          0.1918, -0.0104]], grad_fn=<AddmmBackward>)
'''

# 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_modules` 中。`block`的类型是OrderedDict。
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
'''
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
tensor([[ 0.1602,  0.1257, -0.0031, -0.0120,  0.0065,  0.0318,  0.0542,  0.0386,
          0.0174,  0.2762],
        [ 0.1028,  0.1396, -0.0145,  0.1410,  0.0484,  0.1162,  0.0136, -0.0006,
         -0.0075,  0.0924]], grad_fn=<AddmmBackward>)
'''
# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变。
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及`relu`和`dot`函数。
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
'''
net = FixedHiddenMLP()
net(X)
tensor(0.1178, grad_fn=<SumBackward0>)
'''
# 混合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
'''
tensor(-0.5803, grad_fn=<SumBackward0>)
'''