import torch
#转置

#A = torch.arange(20).reshape(5, 4)
#A.T
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])    
        to
tensor([[ 0,  4,  8, 12, 16],
        [ 1,  5,  9, 13, 17],
        [ 2,  6, 10, 14, 18],
        [ 3,  7, 11, 15, 19]])
        
'''
'''
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B.T

特殊矩阵，转置后相同
'''
#A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
#B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
#print(A * B)
#降维（求和）
A=torch.arange(12).reshape((3,4))
print(A)
print(A.shape, A.sum())
A_sum_axis0 = A.sum(axis=0)#只留下行，列求和
print(A_sum_axis0, A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)#只留下列，行求和
print(A_sum_axis1, A_sum_axis1.shape)
'''
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([3, 4]) tensor(66)
tensor([12, 15, 18, 21]) torch.Size([4])
tensor([ 6, 22, 38]) torch.Size([3])
'''
#点积
'''
y = torch.ones(4, dtype=torch.float32)
x=torch.tensor([0., 1., 2., 3.])
print(torch.dot(x, y))
torch.sum(x * y)#也可达到点乘的结果
torch.mv(A, x)#叉乘
torch.mm(A, x)#矩阵乘
'''

#x1=torch.tensor([[1,2,3],[4,5,6]])
#print(x1.shape)
#x2=torch.tensor([1,2,3])
#print(x2.shape)
#print(torch.mv(x1, x2))
#sum_A = A.sum(axis=1, keepdims=True) 保持纬度不变
# 范数（向量的长度）

u = torch.tensor([3.0, -4.0])
torch.norm(u)
torch.abs(u).sum()#绝对值
#弗罗贝尼乌斯范数 Frobenius norm
#将矩阵平方求和再开方
torch.norm(torch.ones((4, 9)))
'''
tensor(6.)
'''
test=torch.arange(24).reshape((2,3,4))
print(test)
sum1=test.sum(axis=0)
sum2=test.sum(axis=1)
sum3=test.sum(axis=2)
print(sum1,sum1.size())
print(sum2,sum2.size())
print(sum3,sum3.size())