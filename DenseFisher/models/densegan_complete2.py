import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class GenBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(GenBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.ConvTranspose2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.ConvTranspose2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        #print("dense internal shape 1 " + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("dense internal shape 2 " + str(out.size()))
        out = self.conv2(F.relu(self.bn2(out)))
        #print("dense internal shape 3 " + str(out.size()))
        out = torch.cat((x, out), 1)
        #print("dense internal shape 4 " + str(out.size()))
        return out

class GenTransition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(GenTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        
        self.conv1 = nn.ConvTranspose2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.up1 = nn.Upsample(nOutChannels, scale_factor=2, mode='bilinear')

    def forward(self, x):
        #print("transition internal shape 1" + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("transition internal shape 2" + str(out.size()))
        #out = F.avg_pool2d(out, 2)
        out = self.up1(out)
        #print("transition internal shape 3" + str(out.size()))
        return out
    
class GenDenseNet(nn.Module):
    def __init__(self, growthRate, depth, increase, nz, bottleneck=1, verbose=1):
        super(GenDenseNet, self).__init__()
        
        self.verbose = verbose
        self.conv1 = nn.ConvTranspose2d(nz, growthRate*7 , kernel_size=3, padding=1,
                               bias=False)
        self.bn_1 = nn.BatchNorm2d(growthRate*7)
        
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        
        nChannels = growthRate*7
        
        self.dense1 = self._make_dense( nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans1 = GenTransition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans2 = GenTransition( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans3 = GenTransition( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans4 = GenTransition( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans5 = GenTransition( nChannels, nOutChannels)
        
        self.conv_f = nn.ConvTranspose2d(nOutChannels, 3, kernel_size=3, padding=1,
                               bias=False)
        #self.bn_f = nn.BatchNorm2d(nOutChannels)

        #self.bn1 = nn.BatchNorm2d(nChannels)
        self.ufinal = nn.Tanh()
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck=1):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(GenBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print("######################G#####################")
            print("Input shape " + str(x.size()))
            
        out = self.conv1(x)
        
        if self.verbose:
            print("conv1 shape " + str(out.size()))
        out = F.relu(self.bn_1(out))
        out = self.trans1(self.dense1(out))
        if self.verbose:
            print("dense + trans 1 finished shape " + str(out.size()))
        out = self.trans2(self.dense2(out))
        if self.verbose:
            print("dense + trans 2 finished shape " + str(out.size()))
        out = self.trans3(self.dense3(out))
        if self.verbose:
            print("dense + trans 3 finished shape " + str(out.size()))
        out = self.trans4(self.dense4(out))
        if self.verbose:
            print("dense + trans 4 finished shape " + str(out.size()))
        out = self.trans5(self.dense5(out))
        if self.verbose:
            print("dense + trans 5 finished shape " + str(out.size()))
        
        out = self.conv_f(out)
        #out = self.bn_f(out)
        out = self.ufinal(out)
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        #out = F.log_softmax(self.fc(out))
        if self.verbose:
            print("######################G#####################")
        return out
    
    
    

class DisBottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(DisBottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        #print("dense internal shape 1 " + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("dense internal shape 2 " + str(out.size()))
        out = self.conv2(F.relu(self.bn2(out)))
        #print("dense internal shape 3 " + str(out.size()))
        out = torch.cat((x, out), 1)
        #print("dense internal shape 4 " + str(out.size()))
        return out

class DisTransition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(DisTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        #print("transition internal shape 1" + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("transition internal shape 2" + str(out.size()))
        out = F.avg_pool2d(out, 2)
        #print("transition internal shape 3" + str(out.size()))
        return out

#net = densenet.DenseNet(growthRate=12, depth=100, reduction=0.5,
#                            bottleneck=True, nClasses=10)
    
    
class DisDenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, verbose=1, bottleneck=1):
        super(DisDenseNet, self).__init__()

        self.verbose=verbose
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(nChannels)
        
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = DisTransition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = DisTransition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = DisTransition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans4 = DisTransition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans5 = DisTransition(nChannels, nOutChannels)
        
        self.convf = nn.Conv2d(nOutChannels, 1, kernel_size=3, padding=1,
                               bias=False)
        #maybe remove?    
        #self.bnf = nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck=1):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(DisBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print("######################D#####################")
            print("Input shape " + str(x.size()))
         
        out = F.relu(self.conv1(x))
        #out = F.relu(self.bn1(self.conv1(x)))
        
        if self.verbose:
            print("con1 shape " + str(out.size()))
        out = self.trans1(self.dense1(out))
        if self.verbose:
            print("dense + trans 1 finished shape " + str(out.size()))
        out = self.trans2(self.dense2(out))
        if self.verbose:
            print("dense + trans 2 finished shape " + str(out.size()))
        
        out = self.trans3(self.dense3(out))
        if self.verbose:
            print("dense + trans 3 finished shape " + str(out.size()))
        out = self.trans4(self.dense4(out))
        if self.verbose:
            print("dense + trans 4 finished shape " + str(out.size()))
        out = self.trans5(self.dense5(out))
        if self.verbose:
            print("dense + trans 5 finished shape " + str(out.size()))
        out = self.convf(out)
        #out = self.bnf(out)
        if self.verbose:
            print("Final shape " + str(out.size()))
        ##print("dense f finished shape " + str(out.size()))
        ##out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        ##out = F.log_softmax(self.fc(out))
        if self.verbose:
            print("######################D#####################")
        return out.view(-1)
        #return out