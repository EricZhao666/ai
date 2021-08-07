# 不带参数的层
import torch
import torch.nn.functional as F
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
'''
tensor([-2., -1.,  0.,  1.,  2.])
'''
# 放到更复杂的模型里
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
'''
tensor(-3.7253e-09, grad_fn=<MeanBackward0>)
'''
# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 随机获得
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
linear.weight
'''
Parameter containing:
tensor([[ 0.5306, -2.0067,  1.3679],
        [-0.4192, -0.3537, -2.0357],
        [ 1.4975, -1.1942,  0.0699],
        [-1.3198, -1.3796,  0.1917],
        [ 1.0115,  0.5938,  1.5117]], requires_grad=True)
'''
# 进行正向传播计算
linear(torch.rand(2, 5))
'''
tensor([[0.0000, 0.0000, 0.0000],
        [0.5839, 0.0000, 0.0000]])
'''
# 自定义模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
