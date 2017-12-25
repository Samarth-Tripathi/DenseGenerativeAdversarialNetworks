

import torch
import torch.nn as nn
import torch.nn.parallel

import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

    
class DC_netG2_64(nn.Module):
    def __init__(self, isize, nz, ngpu):
        super(DC_netG2_64, self).__init__()
        self.ngpu = ngpu
        
        self.isize = isize
        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        
        
        self.uconv2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.uconv_f2 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(ngf)
        
        self.uconv5 =  nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        out = self.uconv1(input)
        out = self.ubn1(out)
        out = self.ur(out)

        #print (out.size())
        
        out = self.uconv2(out)
        out = self.ubn2(out)
        out = self.ur(out)

        #print (out.size())

        out = self.uconv3(out)
        out = self.ubn3(out)
        out = self.ur(out)
        
        #print (out.size())

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv_f1(out)
        out = self.ubn_f1(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class DC_netD2_64(nn.Module):
    
    def __init__(self, isize, ngpu):
        super(DC_netD2_64, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        self.isize = isize
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv3v2 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn2v2 = nn.BatchNorm2d(ndf * 8)
        self.conv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())

        input = input.resize(input.size()[0], 3, self.isize, self.isize)
        
        out = self.conv1(input)
        out = self.lr(out)
        
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv3v2(out)
        out = self.bn2v2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)  
    
class DC_netD2_better_64(nn.Module):
    
    def __init__(self, isize, ngpu):
        super(DC_netD2_better_64, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        self.isize = isize
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv_f1 = nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn_f1 = nn.BatchNorm2d(ndf)
        
        self.conv_f2 = nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn_f2 = nn.BatchNorm2d(ndf)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv3v2 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn2v2 = nn.BatchNorm2d(ndf * 8)
        self.conv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())

        input = input.resize(input.size()[0], 3, self.isize, self.isize)
        
        out = self.conv1(input)
        out = self.lr(out)

        #print (out.size())
        out = self.conv_f1(out)
        out = self.bn_f1(out)
        out = self.lr(out)
        #print (out.size())
        
        out = self.conv_f2(out)
        out = self.bn_f2(out)
        out = self.lr(out)
        #print (out.size())
        
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv3v2(out)
        out = self.bn2v2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)  
    
    
class Dense_netG3_64(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG3_64, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        
        self.uconv1_d =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn1_d = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2 =  nn.ConvTranspose2d(ngf * 12, ngf * 4, 4, 2, 1, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn2_d = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 6, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d =  nn.ConvTranspose2d(ngf * 2, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn3_d = nn.BatchNorm2d(ngf * 1)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 3,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.uconv_f2 =  nn.ConvTranspose2d(ngf*2, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(ngf)
        
        
        self.uconv5 =  nn.ConvTranspose2d(ngf*3,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv1(input)
        out = self.ubn1(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv1_d(out)
        out = self.ubn1_d(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv2(out)
        out = self.ubn2(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv2_d(out)
        out = self.ubn2_d(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.uconv3(out)
        out = self.ubn3(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv3_d(out)
        out = self.ubn3_d(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        x = out
        
        out = self.uconv_f1(out)
        out = self.ubn_f1(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out
    
    
class Dense_netG_mini_complete_64(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG_mini_complete_64, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        
        self.uconv1_d1 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn1_d1 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv1_d2 =  nn.ConvTranspose2d(ngf * 12, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn1_d2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2 =  nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d1 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn2_d1 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv2_d2 =  nn.ConvTranspose2d(ngf * 6, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn2_d2 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d1 =  nn.ConvTranspose2d(ngf * 2, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn3_d1 = nn.BatchNorm2d(ngf * 1)
        
        self.uconv3_d2 =  nn.ConvTranspose2d(ngf * 3, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn3_d2 = nn.BatchNorm2d(ngf * 1)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 4,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.uconv_f2 =  nn.ConvTranspose2d(ngf*2, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(ngf)
        
        
        self.uconv5 =  nn.ConvTranspose2d(ngf*3,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv1(input)
        out = self.ubn1(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv1_d1(out)
        out = self.ubn1_d1(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv1_d2(out)
        out = self.ubn1_d2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv2(out)
        out = self.ubn2(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv2_d1(out)
        out = self.ubn2_d1(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv2_d2(out)
        out = self.ubn2_d2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.uconv3(out)
        out = self.ubn3(out)
        out = self.ur(out)
        
        x = out
        
        #print (out.size())
        
        out = self.uconv3_d1(out)
        out = self.ubn3_d1(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.uconv3_d2(out)
        out = self.ubn3_d2(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        x = out
        
        out = self.uconv_f1(out)
        out = self.ubn_f1(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        
        #print(out.size())
        #print("**********************G**********************")
        return out
    
    
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
        self.up1 = nn.Upsample(nOutChannels, scale_factor=2, mode='nearest')

    def forward(self, x):
        #print("transition internal shape 1" + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("transition internal shape 2" + str(out.size()))
        #out = F.avg_pool2d(out, 2)
        out = self.up1(out)
        #print("transition internal shape 3" + str(out.size()))
        return out
    
    
class GenDenseNet_64(nn.Module):
    def __init__(self, growthRate, depth, increase, nz, bottleneck=1, verbose=1):
        super(GenDenseNet_64, self).__init__()
        
        self.verbose = verbose
        self.conv1 = nn.ConvTranspose2d(nz, growthRate*10 , kernel_size=3, padding=1,
                               bias=False)
        self.bn_1 = nn.BatchNorm2d(growthRate*10)
        
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        
        nChannels = growthRate*10
        
        self.dense0 = self._make_dense( nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans0 = GenTransition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
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
        
        self.ur = nn.ReLU(True)
        self.ubn_e0 = nn.BatchNorm2d(nOutChannels)
        self.uconv_e1 =  nn.ConvTranspose2d(nOutChannels, nOutChannels, kernel_size=3, padding=1, bias=False)
        self.ubn_e1 = nn.BatchNorm2d(nOutChannels)
        self.uconv_e2 =  nn.ConvTranspose2d(nOutChannels*2, nOutChannels, kernel_size=3, padding=1, bias=False)
        self.ubn_e2 = nn.BatchNorm2d(nOutChannels)
    

        self.bn_f = nn.BatchNorm2d(nOutChannels*3)
        self.conv_f = nn.ConvTranspose2d(nOutChannels*3, 3, kernel_size=3, padding=1,
                               bias=False)
        

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
            
        out = self.trans0(self.dense0(out))
        if self.verbose:
            print("dense + trans 0 finished shape " + str(out.size()))
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
            
        out = self.ubn_e0(out)
        out = self.ur(out)
        
        
        x = out
        
        out = self.uconv_e1(out)
        out = self.ubn_e1(out)
        out = self.ur(out)
        
        if self.verbose:
            print("extra 1 finished shape " + str(out.size()))
        
        out = torch.cat((x, out), 1)
        
        if self.verbose:
            print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.uconv_e2(out)
        out = self.ubn_e2(out)
        out = self.ur(out)
        
        if self.verbose:
            print("extra 2 finished shape " + str(out.size()))
        
        out = torch.cat((x, out), 1)
        
        if self.verbose:
            print ("New size= " + str(out.size()))
        
        out = self.conv_f(F.relu(self.bn_f(out)))
        #out = self.bn_f(out)
        
        if self.verbose:
            print("final finished shape " + str(out.size()))
            
        out = self.ufinal(out)
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        #out = F.log_softmax(self.fc(out))
        if self.verbose:
            print("######################G#####################")
            
        return out
    
    
    
    
class GenBottleneck_alt(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(GenBottleneck_alt, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv2 = nn.ConvTranspose2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        #print("dense internal shape 1 " + str(x.size()))
        out = self.conv2(F.relu(self.bn1(x)))
        #print("dense internal shape 3 " + str(out.size()))
        out = torch.cat((x, out), 1)
        #print("dense internal shape 4 " + str(out.size()))
        return out

class GenTransition_alt(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(GenTransition_alt, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.ConvTranspose2d(nChannels, nOutChannels, 3, 1, 1, 0, bias=False)
        self.up1 = nn.Upsample(nOutChannels, scale_factor=2, mode='nearest')

    def forward(self, x):
        #print("transition internal shape 1" + str(x.size()))
        out = self.conv1(F.relu(self.bn1(x)))
        #print("transition internal shape 2" + str(out.size()))
        #out = F.avg_pool2d(out, 2)
        out = self.up1(out)
        #print("transition internal shape 3" + str(out.size()))
        return out
    
    
class GenDenseNet_64_alt(nn.Module):
    def __init__(self, growthRate, depth, increase, nz, bottleneck=1, verbose=1):
        super(GenDenseNet_64_alt, self).__init__()
        
        self.verbose = verbose
        self.conv1 = nn.ConvTranspose2d(nz, growthRate*10 , kernel_size=3, padding=1,
                               bias=False)
        self.bn_1 = nn.BatchNorm2d(growthRate*10)
        
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        
        nChannels = growthRate*10
        
        self.dense0 = self._make_dense( nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans0 = GenTransition_alt(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense1 = self._make_dense( nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans1 = GenTransition_alt(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans2 = GenTransition_alt( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans3 = GenTransition_alt( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans4 = GenTransition_alt( nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = nChannels-(growthRate*2)
        self.trans5 = GenTransition_alt( nChannels, nOutChannels)
        
        self.ur = nn.ReLU(True)
        self.ubn_e0 = nn.BatchNorm2d(nOutChannels)
        self.uconv_e1 =  nn.ConvTranspose2d(nOutChannels, nOutChannels, kernel_size=3, padding=1, bias=False)
        self.ubn_e1 = nn.BatchNorm2d(nOutChannels)
        self.uconv_e2 =  nn.ConvTranspose2d(nOutChannels*2, nOutChannels, kernel_size=3, padding=1, bias=False)
        self.ubn_e2 = nn.BatchNorm2d(nOutChannels)
    

        self.bn_f = nn.BatchNorm2d(nOutChannels*3)
        self.conv_f = nn.ConvTranspose2d(nOutChannels*3, 3, kernel_size=3, padding=1,
                               bias=False)
        

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
                layers.append(GenBottleneck_alt(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.verbose:
            print("######################G#####################")
            print("Input shape " + str(x.size()))
            
        out = self.conv1(x)
        
        if self.verbose:
            print("conv1 shape " + str(out.size()))
            
        out = self.trans0(self.dense0(out))
        if self.verbose:
            print("dense + trans 0 finished shape " + str(out.size()))
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
            
        out = self.ubn_e0(out)
        out = self.ur(out)
        
        
        x = out
        
        out = self.uconv_e1(out)
        out = self.ubn_e1(out)
        out = self.ur(out)
        
        if self.verbose:
            print("extra 1 finished shape " + str(out.size()))
        
        out = torch.cat((x, out), 1)
        
        if self.verbose:
            print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.uconv_e2(out)
        out = self.ubn_e2(out)
        out = self.ur(out)
        
        if self.verbose:
            print("extra 2 finished shape " + str(out.size()))
        
        out = torch.cat((x, out), 1)
        
        if self.verbose:
            print ("New size= " + str(out.size()))
        
        out = self.conv_f(F.relu(self.bn_f(out)))
        #out = self.bn_f(out)
        
        if self.verbose:
            print("final finished shape " + str(out.size()))
            
        out = self.ufinal(out)
        #out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        #out = F.log_softmax(self.fc(out))
        if self.verbose:
            print("######################G#####################")
            
        return out