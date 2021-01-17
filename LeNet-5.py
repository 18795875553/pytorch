import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os

EPOCH = 20
BATCH_SIZE = 10
LR = 0.001  
DOWNLOAD_MNIST = False

#-------------------------------数据集---------------------------------

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # training data
    transform=torchvision.transforms.ToTensor(),   
    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
 
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
 

#-------------------------------网络结构---------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # 上述是自定义网络的常规写法
        self.conv1 = nn.Sequential(        
            nn.Conv2d(1, 6, 5), # 输入通道，输出通道，卷积核大小
            nn.ReLU(),                 
            nn.MaxPool2d(2), 
        )
        self.conv2 = nn.Sequential(        
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(),                 
            nn.MaxPool2d(2), 
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 120), # 输入特征，输出特征
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2.view(x.size(0), -1) # 展开成一维向量，方便后面进行全连接
        x3 = self.fc1(x2)
        x4 = self.fc2(x3)
        x5 = self.fc3(x4)
        return torch.nn.functional.log_softmax(x5, dim=1)

net = Net()
print(net)

#------------------------------------开始训练-----------------------------------

loss_func = nn.CrossEntropyLoss()   # 损失函数
optimizer = torch.optim.Adam(net.parameters(),lr = LR) # 梯度下降

cuda_gpu = torch.cuda.is_available() # have gpu
for epoch in range(EPOCH):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda_gpu:
            data, target = data.cuda(), target.cuda()
            net.cuda()

        output = net(data) # 网络输出结果

        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 400 == 0:
            #--------------------------test-------------------------
            net.eval()
            correct = 0
            for data, target in test_loader:
                if cuda_gpu:
                    data, target = data.cuda(), target.cuda()
                    net.cuda()

                output = net(data)
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
        
            accuracy = 1. * correct / len(test_loader.dataset)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)  
