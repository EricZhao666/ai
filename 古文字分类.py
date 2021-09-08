# 读取照片
import torch
from torchvision import transforms
from d2l import torch as d2l
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# 图片地址
ROOT_TRAIN = r'F:\train'
ROOT_TEST = r'F:\test'
# 灰度图
train_transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(1),])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Grayscale(1)])
# 读入图片
train_dataset = ImageFolder(ROOT_TRAIN,transform=train_transform)
test_dataset = ImageFolder(ROOT_TEST,transform=train_transform)
print(len(train_dataset))
print(len(test_dataset))
# 模型
net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过度拟合
    nn.Linear(256, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 5))
# 加载训练数据
tr_batch_size=512
te_batch_size=512
train_dataloader = DataLoader(train_dataset, batch_size=tr_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=te_batch_size,shuffle=True)
# 评估准确率
def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)。"""
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 判断是否在gpu上跑
    print('training on', device)
    net.to(device)
    # 优化函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        # 累加器
        metric = d2l.Accumulator(3)
        net.train()
        print("开始训练")
        for  i,(X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print("第",epoch,"训练结果")
        print("损失")
        print(train_l)
        print("训练准确率")
        print(train_acc)
        print("测试准确率")
        print(test_acc)
lr, num_epochs, batch_size = 0.0001, 20, 512
train_ch6(net,train_dataloader,test_dataloader,num_epochs,lr,d2l.try_gpu())