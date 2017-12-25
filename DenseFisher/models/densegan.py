

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

    
    
class Dense_netG3(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG3, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d =  nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d = nn.BatchNorm2d(ngf * 2)
        
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
        
        out = self.uconv2(input)
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
        x=out
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out
    
class Dense_netG6(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG6, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d1 =  nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d1 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 12, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d1 =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d1 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d2 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d2 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 6,     ngf, 4, 2, 1, 0, bias=False)
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
        
        out = self.uconv2(input)
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
        x=out
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out

    
class Dense_netG9(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG9, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d1 =  nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d1 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d3 =  nn.ConvTranspose2d(ngf * 12, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d3 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 16, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d1 =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d1 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d2 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d2 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d3 =  nn.ConvTranspose2d(ngf * 6, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 8,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.uconv_f2 =  nn.ConvTranspose2d(ngf*2, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(ngf)
        
        self.uconv_f3 =  nn.ConvTranspose2d(ngf*3, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f3 = nn.BatchNorm2d(ngf)
        
        self.uconv5 =  nn.ConvTranspose2d(ngf*4,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv2(input)
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
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv2_d3(out)
        out = self.ubn2_d3(out)
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
        
        x = out
        
        out = self.uconv3_d3(out)
        out = self.ubn3_d3(out)
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
        
        x=out
        
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x=out
        
        out = self.uconv_f3(out)
        out = self.ubn_f3(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out
    

class Dense_netG9_r(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG9_r, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d1 =  nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d1 = nn.BatchNorm2d(ngf * 4)
        
        self.ubn2_d2p = nn.BatchNorm2d(ngf * 8)
        self.uconv2_d2 =  nn.ConvTranspose2d(ngf * 8, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn2_d2 = nn.BatchNorm2d(ngf * 2)
        
        self.ubn2_d3p = nn.BatchNorm2d(ngf * 10)
        self.uconv2_d3 =  nn.ConvTranspose2d(ngf * 10, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn2_d3 = nn.BatchNorm2d(ngf * 1)
        
        self.ubn3p = nn.BatchNorm2d(ngf * 11)
        self.uconv3 =  nn.ConvTranspose2d(ngf * 11, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d1 =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d1 = nn.BatchNorm2d(ngf * 2)
        
        self.ubn3_d2p = nn.BatchNorm2d(ngf * 4)
        self.uconv3_d2 =  nn.ConvTranspose2d(ngf * 4, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn3_d2 = nn.BatchNorm2d(ngf * 1)
        
        self.ubn3_d3p = nn.BatchNorm2d(ngf * 5)
        self.uconv3_d3 =  nn.ConvTranspose2d(ngf * 5, 32, 3, 1, 1, 0, bias=False)
        self.ubn3_d3 = nn.BatchNorm2d(32)
        
        self.ubn4p = nn.BatchNorm2d(352)
        self.uconv4 =  nn.ConvTranspose2d(352,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.ubn_f2p = nn.BatchNorm2d(ngf*2)
        self.uconv_f2 =  nn.ConvTranspose2d(ngf*2, 32, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(32)
        
        self.ubn_f3p = nn.BatchNorm2d(160)
        self.uconv_f3 =  nn.ConvTranspose2d(160, 16, 3, 1, 1, 0, bias=False)
        self.ubn_f3 = nn.BatchNorm2d(16)
        
        self.ubn5p = nn.BatchNorm2d(176)
        self.uconv5 =  nn.ConvTranspose2d(176, nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv2(input)
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
        
        out = self.ubn2_d2p(out)
        out = self.uconv2_d2(out)
        out = self.ubn2_d2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn2_d3p(out)
        out = self.uconv2_d3(out)
        out = self.ubn2_d3(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn3p(out)
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
        
        out = self.ubn3_d2p(out)
        out = self.uconv3_d2(out)
        out = self.ubn3_d2(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.ubn3_d3p(out)
        out = self.uconv3_d3(out)
        out = self.ubn3_d3(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.ubn4p(out)
        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv_f1(out)
        out = self.ubn_f1(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x=out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn_f2p(out)
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x=out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn_f3p(out)
        out = self.uconv_f3(out)
        out = self.ubn_f3(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn5p(out)
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out
    
    
class Dense_netG12_r(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG12_r, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv2_d1 =  nn.ConvTranspose2d(ngf * 4, ngf * 4, 3, 1, 1, 0, bias=False)
        self.ubn2_d1 = nn.BatchNorm2d(ngf * 4)
        
        self.ubn2_d2p = nn.BatchNorm2d(ngf * 8)
        self.uconv2_d2 =  nn.ConvTranspose2d(ngf * 8, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn2_d2 = nn.BatchNorm2d(ngf * 2)
        
        self.ubn2_d3p = nn.BatchNorm2d(ngf * 10)
        self.uconv2_d3 =  nn.ConvTranspose2d(ngf * 10, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn2_d3 = nn.BatchNorm2d(ngf * 1)
        
        self.ubn2_d4p = nn.BatchNorm2d(ngf * 11)
        self.uconv2_d4 =  nn.ConvTranspose2d(ngf * 11, 32, 3, 1, 1, 0, bias=False)
        self.ubn2_d4 = nn.BatchNorm2d(32)
        
        self.ubn3p = nn.BatchNorm2d(736)
        self.uconv3 =  nn.ConvTranspose2d(736, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d1 =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d1 = nn.BatchNorm2d(ngf * 2)
        
        self.ubn3_d2p = nn.BatchNorm2d(ngf * 4)
        self.uconv3_d2 =  nn.ConvTranspose2d(ngf * 4, ngf * 1, 3, 1, 1, 0, bias=False)
        self.ubn3_d2 = nn.BatchNorm2d(ngf * 1)
        
        self.ubn3_d3p = nn.BatchNorm2d(ngf * 5)
        self.uconv3_d3 =  nn.ConvTranspose2d(ngf * 5, 32, 3, 1, 1, 0, bias=False)
        self.ubn3_d3 = nn.BatchNorm2d(32)
        
        self.ubn3_d4p = nn.BatchNorm2d(352)
        self.uconv3_d4 =  nn.ConvTranspose2d(352, 16, 3, 1, 1, 0, bias=False)
        self.ubn3_d4 = nn.BatchNorm2d(16)
        
        self.ubn4p = nn.BatchNorm2d(368)
        self.uconv4 =  nn.ConvTranspose2d(368,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        
        self.uconv_f1 =  nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, 0, bias=False)
        self.ubn_f1 = nn.BatchNorm2d(ngf)
        
        self.ubn_f2p = nn.BatchNorm2d(ngf*2)
        self.uconv_f2 =  nn.ConvTranspose2d(ngf*2, 32, 3, 1, 1, 0, bias=False)
        self.ubn_f2 = nn.BatchNorm2d(32)
        
        self.ubn_f3p = nn.BatchNorm2d(160)
        self.uconv_f3 =  nn.ConvTranspose2d(160, 16, 3, 1, 1, 0, bias=False)
        self.ubn_f3 = nn.BatchNorm2d(16)
        
        self.ubn_f4p = nn.BatchNorm2d(176)
        self.uconv_f4 =  nn.ConvTranspose2d(176, 8, 3, 1, 1, 0, bias=False)
        self.ubn_f4 = nn.BatchNorm2d(8)
        
        self.ubn5p = nn.BatchNorm2d(184)
        self.uconv5 =  nn.ConvTranspose2d(184, nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv2(input)
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
        
        out = self.ubn2_d2p(out)
        out = self.uconv2_d2(out)
        out = self.ubn2_d2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn2_d3p(out)
        out = self.uconv2_d3(out)
        out = self.ubn2_d3(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn2_d4p(out)
        out = self.uconv2_d4(out)
        out = self.ubn2_d4(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn3p(out)
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
        
        out = self.ubn3_d2p(out)
        out = self.uconv3_d2(out)
        out = self.ubn3_d2(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.ubn3_d3p(out)
        out = self.uconv3_d3(out)
        out = self.ubn3_d3(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x = out
        
        out = self.ubn3_d4p(out)
        out = self.uconv3_d4(out)
        out = self.ubn3_d4(out)
        out = self.ur(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.ubn4p(out)
        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        x = out
        
        #print ("New size= " + str(out.size()))
        
        out = self.uconv_f1(out)
        out = self.ubn_f1(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x=out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn_f2p(out)
        out = self.uconv_f2(out)
        out = self.ubn_f2(out)
        out = self.ur(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        x=out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn_f3p(out)
        out = self.uconv_f3(out)
        out = self.ubn_f3(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x=out
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn_f4p(out)
        out = self.uconv_f4(out)
        out = self.ubn_f4(out)
        out = self.ur(out)
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.ubn5p(out)
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out
    
    
    
class Dense_netD3(nn.Module):
    
    def __init__(self, ngpu):
        super(Dense_netD3, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv_f1 = nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn_f1 = nn.BatchNorm2d(ndf)
        
        self.conv_f2 = nn.Conv2d(ndf*2, ndf, 3, 1, 1, bias=False)
        self.bn_f2 = nn.BatchNorm2d(ndf)
        
        self.conv2 = nn.Conv2d(ndf*3, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        
        self.conv2_d = nn.Conv2d(ndf*2, ndf * 2, 3, 1, 1, bias=False)
        self.bn1_d = nn.BatchNorm2d(ndf * 2)
        
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        
        self.conv3_d = nn.Conv2d(ndf*4, ndf * 4, 3, 1, 1, bias=False)
        self.bn2_d = nn.BatchNorm2d(ndf * 4)
        
        self.conv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        
        out = self.conv1(input)
        out = self.lr(out)
        
        #print (out.size())
        
        x = out
        
        out = self.conv_f1(out)
        out = self.bn_f1(out)
        out = self.lr(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        x=out
        
        out = self.conv_f2(out)
        out = self.bn_f2(out)
        out = self.lr(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())
        
        x=out
        
        out = self.conv2_d(out)
        out = self.bn1_d(out)
        out = self.lr(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)
        
        #print (out.size())
        
        x=out
        
        out = self.conv3_d(out)
        out = self.bn2_d(out)
        out = self.lr(out)
        
        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)   

    
class DC_netG2(nn.Module):
    def __init__(self, nz, ngpu):
        super(DC_netG2, self).__init__()
        self.ngpu = ngpu

        nc = 3
        ngf = 64
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
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
        #out = self.uconv1(input)
        #out = self.ubn1(out)
        #out = self.ur(out)

        ##print (out.size())
        
        out = self.uconv2(input)
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


class DC_netD2(nn.Module):
    
    def __init__(self, isize, ngpu):
        super(DC_netD2, self).__init__()
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
        self.conv4 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)

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
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)   