import torch
x = torch.arange(4.0, requires_grad=True)
print(torch.dot(x, x))
y = 2 * torch.dot(x, x)
y.backward()
x.grad# tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x# tensor([True, True, True, True])

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad# tensor([1., 1., 1., 1.])

# 对非标量调用`backward`需要传入一个`gradient`参数，该参数指定微分函数关于`self`的梯度。
# 在我们的例子中，我们只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad# tensor([0., 2., 4., 6.])

#
x.grad.zero_()
y = x * x
u = y.detach()#该函数用于将y认为一个常数
z = u * x
#求导
z.sum().backward()
x.grad == u#tensor([True, True, True, True])
#但是y任然可以计算
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x#tensor([True, True, True, True])

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
a.grad == d / a#tensor(True)
'''
我们现在可以分析上面定义的 f 函数。
请注意，它在其输入 a 中是分段线性的。换言之，对于任何 a，存在某个常量标量 k，使得 f(a) = k * a，其中 k 的值取决于输入 a。
因此，d / a 允许我们验证梯度是否正确。
'''