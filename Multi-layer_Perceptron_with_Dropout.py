import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

transform=transforms.ToTensor()
train_data=datasets.FashionMNIST(root='/data',train=True,transform=transform,download=True)
test_data=datasets.FashionMNIST(root='/data',train=True,transform=transform,download=True)

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=1000)

class MLPDropout(nn.Module):
    def __init__ (self):
        super(MLPDropout,self).__init__()
        self.fc1=nn.Linear(28*28,256)
        self.drop1=nn.Dropout(0.5)
        self.fc2=nn.Linear(256,128)
        self.drop2=nn.Dropout(0.5)
        self.out=nn.Linear(128,10)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=F.relu(self.fc1(x))
        x=self.drop1(x)
        x=F.relu(self.fc2(x))
        x=self.drop2(x)
        x=self.out(x)
        return x
model=MLPDropout()

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(5):
    for images,labels in train_loader:
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"the loss is {loss:.4f} in epoch {epoch+1}")

correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    print(f"correctnes is {(correct/total):.4f}")
