import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
# 输出最后一层的参数
print(net[2].state_dict())
# 直接访问准确数据
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
net[2].weight.grad == None
# True 还没计算没有梯度
# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
net.state_dict()['2.bias'].data
'''
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
tensor([0.0773])
'''
# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
'''
tensor([[0.1578],
        [0.1578]], grad_fn=<AddmmBackward>)
'''
# 内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)   # 更新权重
        nn.init.zeros_(m.bias)   # 偏置为0

net.apply(init_normal)   # 应用
net[0].weight.data[0], net[0].bias.data[0]
'''
(tensor([-0.0174, -0.0015, -0.0094, -0.0061]), tensor(0.))
'''
# 可以初始化为常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
'''
(tensor([1., 1., 1., 1.]), tensor(0.))
'''
# 不同块的初始化
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
'''
tensor([0.0011, 0.1110, 0.5228, 0.5800])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
'''
# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
'''
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[ 0.0000, -7.1017,  8.6895,  8.3252],
        [ 7.3297,  0.0000,  0.0000, -0.0000]], grad_fn=<SliceBackward>)
'''
# 也可以直接替换
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
'''
tensor([42.0000, -6.1017,  9.6895,  9.3252])
'''
# 参数绑定,在多个层间共享参数
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值。
print(net[2].weight.data[0] == net[4].weight.data[0])
'''
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
这个例子表明第二层和第三层的参数是绑定的。它们不仅值相等，而且由相同的张量表示。
因此，如果我们改变其中一个参数，另一个参数也会改变。你可能会想，当参数绑定时，梯度会发生什么情况？
答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层和第三个隐藏层的梯度会加在一起。
'''