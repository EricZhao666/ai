# 加载和保存张量
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
'''
tensor([0, 1, 2, 3])
'''
# 存储list
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
'''
(tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
'''
# 我们甚至可以写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
'''
{'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
'''
# 加载和保存模型参数
class MLP(nn.Module):  # 多层感知机
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 存储到字典
torch.save(net.state_dict(), 'mlp.params')
# 加载参数
clone = MLP()  # 模型也要复制
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
'''
MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
'''
# 验证
Y_clone = clone(X)
Y_clone == Y
'''
tensor([[True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True]])
'''