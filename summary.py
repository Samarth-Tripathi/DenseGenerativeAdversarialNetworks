
from torch.nn.modules.module import _addindent
import torch
import numpy as np
import torch.nn as nn
def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr
        
class _netG(nn.Module):
    
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        
        self.uconv1 =  nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ngf * 8)
        self.ur = nn.ReLU(True)
        self.uconv2 =  nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.ubn2 = nn.BatchNorm2d(ngf * 4)
        self.uconv3 =  nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.ubn3 = nn.BatchNorm2d(ngf * 2)
        self.uconv4 =  nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False)
        self.ubn4 = nn.BatchNorm2d(ngf)
        self.uconv4 =  nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, bias=False)
        self.ufinal = nn.Tanh()


    def forward(self, input):
        
        out = self.uconv1(input)
        out = self.ubn1(out)
        out = self.ur(out)
        x = out
        out = self.uconv2(out)
        out = self.ubn2(out)
        out = self.ur(out)
        y = out
        out = torch.cat((x, out), 1)
        x = y
        out = self.uconv3(out)
        out = self.ubn3(out)
        out = self.ur(out)
        y = out
        out = torch.cat((x, out), 1)
        x = y
        out = self.uconv4(out)
        out = self.ubn4(out)
        out = self.ur(out)
        out = torch.cat((x, out), 1)
        
        out = self.conv5(out)
        out = self.final(out)
        
        #output = self.make_main(input)
        return out.view(-1, 1).squeeze(1)



class _netD(nn.Module):
    
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        nc = 3
        ndf = 64
        
        self.conv1 =  nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lr = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        self.final = nn.Sigmoid()


    def forward(self, input):
        
        out = self.conv1(input)
        out = self.lr(out)
        x = out
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lr(out)
        out = torch.cat((x, out), 1)
        x = out
        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lr(out)
        out = torch.cat((x, out), 1)
        x = out
        out = self.conv4(out)
        out = self.bn3(out)
        out = self.lr(out)
        out = torch.cat((x, out), 1)

        out = self.conv5(out)
        out = self.final(out)
        
        #output = self.make_main(input)
        return out.view(-1, 1).squeeze(1)

# Test
import densenet
#import densenet_notransition
#model = densenet_notransition.DenseNet(12,50,0.5,10,1,4)
model = densenet.DenseNet(12,50,0.5,10,1)
#model = _netG(0)
print(torch_summarize(model))
