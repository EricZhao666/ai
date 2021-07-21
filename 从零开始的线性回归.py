import matplotlib
import random
import torch
from d2l import torch as d2l
# 生成数据集
def synthetic_data(w, b, num_examples):  #@save
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)   # 设置随机误差
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
'''
(**我们使用线性模型参数 w=[2,−3.4]⊤w=[2,−3.4]⊤ 、 b=4.2b=4.2 和噪声项 ϵϵ 生成数据集及其标签：
y=Xw+b+ϵ.
**)
'''
print('features:', features[0], '\nlabel:', labels[0])
# 画图
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(),
                labels.detach().numpy(), 1);
# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))# 位置下标
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)    # 打乱下标
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i +batch_size, num_examples)])    #   读取从0到我们想要的大小
        yield features[batch_indices], labels[batch_indices]    # yield 返回数据
# 测试获取
batch_size = 10
'''
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
    '''
# 初始化模型参数
'''
权重向量w和偏置b
'''
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)    # normal为正态分布，均值为0，方差0.01
b = torch.zeros(1, requires_grad=True)
#定义模型
def linreg(X, w, b):  #@save
    """线性回归模型。"""
    # y = Xw + b
    return torch.matmul(X, w) + b
'''
matmul为矩阵乘以向量
'''
# 定义损失函数
'''
l(i)(w,b)=12(ŷ (i)−y(i))/2    ŷ (i)为预测值  y(i)为真实值
'''
def squared_loss(y_hat, y):  #@save
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2
# 定义优化算法
'''
'''
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size    # 之前损失没求平均值，在这里补上
            param.grad.zero_()    # 梯度清零


# 接下来为训练过程
#定义超参数
lr = 1   # 学习率 梯度下降的速度，不能太小也不能太大
num_epochs = 3     # 将数据过三遍
net = linreg    # 模型名称
loss = squared_loss    # 均方损失
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()    # 向量求和转标量算梯度
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

'''
比较真实值和预测值
'''
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')