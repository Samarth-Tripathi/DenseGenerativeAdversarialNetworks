

import torch
import torch.nn as nn
import torch.nn.parallel

'''
class Dense_netG(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 3, 2, 0, 1, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, 1, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        self.uconv2_d =  nn.ConvTranspose2d(ngf * 4, ngf * 8, 3, 1, 1, 0, bias=False)
        self.ubn2_d = nn.BatchNorm2d(ngf * 8)
        self.uconv3 =  nn.ConvTranspose2d(ngf * 12, ngf * 2, 3, 2, 1, 1, bias=False)

        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 2,     ngf, 3, 2, 1, 1, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        self.uconv5 =  nn.ConvTranspose2d(ngf,     nc, 3, 2, 1, 1, bias=False)
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
        #print (out.size())

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class Dense_netD(nn.Module):
    
    def __init__(self, ngpu):
        super(Dense_netD, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        
        self.conv1 =  nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv3_d = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=False)
        self.bn2_d = nn.BatchNorm2d(ndf * 8)
        self.conv4 = nn.Conv2d(ndf * 12, ndf * 8, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 3, 2, 0, bias=False)
        self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        #input = input.resize(int(input.size()[0]*input.size()[1]*input.size()[2]*input.size()[3]/(3*opt.imageSize*opt.imageSize)), 3, opt.imageSize, opt.imageSize)
        
        
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 64, 64)
        
        out = self.conv1(input)
        out = self.lr(out)

        #print (out.size())
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)
        
        x = out

        #print (out.size())
        out = self.conv3_d(out)
        out = self.bn2_d(out)
        out = self.lr(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        out = self.conv4(out)
        out = self.bn3(out)
        out = self.lr(out)

        #print (out.size())
        out = self.conv5(out)
        #print (out.size())
        out = self.final(out)
        
        
        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1, 1).squeeze(1)
'''    
    
class Dense_netG(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 32
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 4,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
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
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class Dense_netD(nn.Module):
    
    def __init__(self, ngpu):
        super(Dense_netD, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 32
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        
        self.conv3_d = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.bn2_d = nn.BatchNorm2d(ndf * 2)
        
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        #input = input.resize(int(input.size()[0]*input.size()[1]*input.size()[2]*input.size()[3]/(3*opt.imageSize*opt.imageSize)), 3, opt.imageSize, opt.imageSize)
        
        
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 32, 32)
        
        out = self.conv1(input)
        out = self.lr(out)

        #print (out.size())
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())
        
        x = out

        #print (out.size())
        out = self.conv3_d(out)
        out = self.bn2_d(out)
        out = self.lr(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv4(out)
        out = self.bn3(out)
        out = self.lr(out)

        #print (out.size())
        out = self.conv5(out)
        #print (out.size())
        #out = self.final(out)
        
        
        #print (out.size())
        #print("**********************D**********************")

        #return out.view(-1, 1).squeeze(1)
        return out.view(-1)
    
class Dense_netG2(nn.Module):
    def __init__(self, nz, ngpu):
        super(Dense_netG2, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 64
        
        
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv3_d =  nn.ConvTranspose2d(ngf * 2, ngf * 2, 3, 1, 1, 0, bias=False)
        self.ubn3_d = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 4,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        self.uconv5 =  nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        
        out = self.uconv2(input)
        out = self.ubn2(out)
        out = self.ur(out)

        #print (out.size())

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
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class Dense_netD2(nn.Module):
    
    def __init__(self, ngpu):
        super(Dense_netD2, self).__init__()
        self.ngpu = ngpu
        nc = 3     
        
        ndf = 64
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        
        self.conv3_d = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.bn2_d = nn.BatchNorm2d(ndf * 2)
        
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)



    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 32, 32)
        
        out = self.conv1(input)
        out = self.lr(out)

        #print (out.size())
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())
        
        x = out

        out = self.conv3_d(out)
        out = self.bn2_d(out)
        out = self.lr(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)

        #print (out.size())
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)
    
    
    
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
        self.uconv5 =  nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, 0, bias=False)
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
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        
        self.conv3_d = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=False)
        self.bn2_d = nn.BatchNorm2d(ndf * 2)
        
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        
        self.conv4_d = nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False)
        self.bn3_d = nn.BatchNorm2d(ndf * 4)
        
        self.conv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)



    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 32, 32)
        
        out = self.conv1(input)
        out = self.lr(out)

        #print (out.size())
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        #print (out.size())
        
        x = out

        out = self.conv3_d(out)
        out = self.bn2_d(out)
        out = self.lr(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)
        
        x = out
        
        #print (out.size())
        
        out = self.conv4_d(out)
        out = self.bn3_d(out)
        out = self.lr(out)

        #print (out.size())
        
        out = torch.cat((x, out), 1)
        
        #print ("New size= " + str(out.size()))
        
        out = self.conv4(out)

        #print (out.size())
        #print("**********************D**********************")

        return out.view(-1)
    
    
    
    
class DC_netG(nn.Module):
    def __init__(self, nz, ngpu):
        super(DC_netG, self).__init__()
        self.ngpu = ngpu
        
        nc = 3
        ngf = 32
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 1, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        
        self.uconv3 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        
        self.uconv4 =  nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, 0, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
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
        
        x = out
        
        #print (out.size())

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class DC_netD(nn.Module):
    
    def __init__(self, ngpu):
        super(DC_netD, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 32
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False)
        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        #input = input.resize(int(input.size()[0]*input.size()[1]*input.size()[2]*input.size()[3]/(3*opt.imageSize*opt.imageSize)), 3, opt.imageSize, opt.imageSize)
        
        
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 32, 32)
        
        out = self.conv1(input)
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
        out = self.bn3(out)
        out = self.lr(out)

        #print (out.size())
        out = self.conv5(out)
        
        
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
        self.uconv5 =  nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, 0, bias=False)
        self.ufinal = nn.Tanh()
        
    def forward(self, input):
        
        #print("**********************G**********************")
        #print (input.size())
        #out = self.uconv1(input)
        #out = self.ubn1(out)
        #out = self.ur(out)

        #print (out.size())
        
        out = self.uconv2(input)
        out = self.ubn2(out)
        out = self.ur(out)

        #print (out.size())

        out = self.uconv3(out)
        out = self.ubn3(out)
        out = self.ur(out)
        
        #x = out
        
        #print (out.size())

        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        #print (out.size())
        
        out = self.uconv5(out)
        #print (out.size())
        out = self.ufinal(out)
        

        #print(out.size())
        #print("**********************G**********************")
        return out


class DC_netD2(nn.Module):
    
    def __init__(self, ngpu):
        super(DC_netD2, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)

        #self.final = nn.Sigmoid()


    def forward(self, input):
        
        #print("**********************D**********************")
        #print (input.size())
        #input = input.resize(int(input.size()[0]*input.size()[1]*input.size()[2]*input.size()[3]/(3*opt.imageSize*opt.imageSize)), 3, opt.imageSize, opt.imageSize)
        
        
        
        #input = input.resize(input.size()[0], 3, opt.imageSize, opt.imageSize)
        
        #Hard coding for cifar
        input = input.resize(input.size()[0], 3, 32, 32)
        
        out = self.conv1(input)
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