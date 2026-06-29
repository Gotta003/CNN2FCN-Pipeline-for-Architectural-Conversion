from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherCNN(nn.Module):
    def __init__(self, input_shape=(1, 40, 40), num_classes=11):
        super().__init__()
        C=input_shape[0]
        #Conv Block 1
        self.conv1=nn.Conv2d(C, 32, 3, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(32)
        self.pool1=nn.MaxPool2d(2, 2)
        self.drop1=nn.Dropout(0.2)
        #Conv Block 2
        self.conv2=nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(64)
        self.pool2=nn.MaxPool2d(2, 2)
        self.drop2=nn.Dropout(0.3)
        #Conv Block 3
        self.conv3=nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3=nn.BatchNorm2d(128)
        self.pool3=nn.MaxPool2d(2, 2)
        self.drop3=nn.Dropout(0.4)
        #Fully Connected
        self.fc1=nn.Linear(128*5*5, 256, bias=False)
        self.bn4=nn.BatchNorm1d(256)
        self.drop4=nn.Dropout(0.5)
        self.fc2=nn.Linear(256, 128, bias=False)
        self.bn5=nn.BatchNorm1d(128)
        self.drop5=nn.Dropout(0.5)
        self.fc3=nn.Linear(128, 64, bias=False)
        self.bn6=nn.BatchNorm1d(64)
        self.drop6=nn.Dropout(0.5)
        self.fc_out=nn.Linear(64, num_classes)
        self._hints=[None]*6
        
    def forward(self, x):
        x=self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        self._hints[0]=x.detach()
        x=self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        self._hints[1]=x.detach()
        x=self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        self._hints[2]=x.detach()
        x=x.flatten(1)
        x=self.drop4(F.relu(self.bn4(self.fc1(x))))
        self._hints[3]=x.detach()
        x=self.drop5(F.relu(self.bn5(self.fc2(x))))
        self._hints[4]=x.detach()
        x=self.drop6(F.relu(self.bn6(self.fc3(x))))
        self._hints[5]=x.detach()
        return self.fc_out(x)
    
    def get_hints(self):
        return [h.view(h.size(0), -1) for h in self._hints]