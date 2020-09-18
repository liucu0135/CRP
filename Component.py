import torch.nn as nn
import torch
import math
import numpy as np

class dense_unit(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, kernel=3,pad=1, bn=False, IN=False):
        super(dense_unit, self).__init__()
        layers=[]
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(in_ch,out_ch,kernel,1,pad))
        # layers.append(nn.Dropout(0.2))
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if IN:
            layers.append(nn.InstanceNorm2d(out_ch))
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x = self.layers(input)
        return x

class dense_unitT(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, kernel=3,pad=1, bn=False, IN=False):
        super(dense_unitT, self).__init__()
        layers=[]
        # layers.append(nn.Dropout(0.2))
        layers.append(nn.ConvTranspose2d(in_ch,out_ch,kernel,stride=1, padding=pad))
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if IN:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*layers)
    def forward(self, input):
        x = self.layers(input)
        return x

class Dense_block(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, bn=False, IN=False):
        super(Dense_block, self).__init__()
        self.d1 = dense_unit(in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN)
        self.d2 = dense_unit(in_ch=out_ch+in_ch,out_ch=out_ch, bn=bn, IN=IN)
    def forward(self, input):
        x1 = self.d1(input)
        x2 = torch.cat((x1,input),dim=1)
        x3 = self.d2(x2)
        return torch.cat((input,x1,x3),dim=1)


class Dense_blockT(nn.Module):
    def __init__(self, in_ch=16, out_ch=16, IN=False, bn=False):
        super(Dense_blockT, self).__init__()
        self.inch=in_ch
        self.outch=out_ch
        self.d1 = dense_unitT(in_ch=in_ch,out_ch=out_ch, bn=bn, IN=IN)
        self.d2 = dense_unitT(in_ch=in_ch,out_ch=out_ch+in_ch, bn=bn, IN=IN)
    def forward(self, input):
        x1=input[:,:self.outch,:,:]
        x2=input[:,self.outch:(self.outch+self.inch),:,:]
        # print(input.shape, self.outch, self.inch)
        x3=input[:,(self.outch+self.inch):,:,:]
        x3 = self.d2(x3)
        x31=x3[:,:self.inch,:,:]
        x2=self.d1(x31+x2)
        x3=x3[:,self.inch:,:,:]
        return x1+x2+x3

class Transition_blockT(nn.Module):
    def __init__(self, in_ch=16, out_ch=48, IN=False):
        super(Transition_blockT, self).__init__()
        self.d1 = nn.ConvTranspose2d(in_ch,in_ch, 2,2,0)
        # self.bn = nn.BatchNorm2d(in_ch)
        self.d2 = dense_unitT(in_ch=in_ch,out_ch=out_ch, kernel=1, pad=0, IN=IN)
    def forward(self, input):
        x1 = self.d1(input)
        # x1 = self.bn(x1)
        x1 = self.d2(x1)
        return x1



class Transition_block(nn.Module):
    def __init__(self, in_ch=16, out_ch=48, bn=False, IN=False):
        super(Transition_block, self).__init__()
        # self.d1 = dense_unit(in_ch=in_ch,out_ch=out_ch, kernel=1, pad=0, bn=bn)
        # # self.d2 = nn.AvgPool2d(2,2)
        # self.d2 = nn.Conv2d(out_ch, out_ch, 2, 2, 0)
        # self.bn=nn.BatchNorm2d(out_ch)
        layers=[
            dense_unit(in_ch=in_ch, out_ch=out_ch, kernel=1, pad=0, bn=bn, IN=IN),
            nn.AvgPool2d(2,2),
            # nn.Conv2d(out_ch, out_ch, 2, 2, 0),

        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        self.layers=nn.Sequential(*layers)

    def forward(self, input):
        x1 = self.layers(input)
        return x1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv_base = [
            nn.LayerNorm((1,32,32)),
            nn.Conv2d(1, 32, 3, 1, 1),
            Dense_block(32, 32),
            Transition_block(32 * 2 + 32, 64),
        ]

        conv_base2 = [
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            Dense_block(32, 32, bn=True),
            Transition_block(32 * 2 + 32, 64, bn=True),
        ]
        conv_normal = [
            Dense_block(64, 16),
            nn.Conv2d(16 * 2 + 64, 64, 1, 1, 0),  # 64x16x16
            Transition_block(64, 64),
            Dense_block(64, 16),
            nn.Conv2d(16 * 2 + 64, 64, 1, 1, 0),  # 64x8x8
        ]
        conv_material = [
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        fcm1=[
            nn.Linear(64*8*8,512),
            nn.ReLU(),
        ]

        fcn1=[
            nn.Linear(64*8*8,512),
            nn.LeakyReLU(),
        ]

        self.conv_base = nn.Sequential(*conv_base)
        self.conv_base2 = nn.Sequential(*conv_base2)
        self.conv_normal = nn.Sequential(*conv_normal)
        self.conv_material = nn.Sequential(*conv_material)
        # self.conv_material2 = nn.Sequential(*conv_material2)
        self.fcm1 = nn.Sequential(*fcm1)
        self.fcn1 = nn.Sequential(*fcn1)
    def forward(self, input):


        x = self.conv_base(input)
        x2 = self.conv_base2(input)

        m = self.conv_material(x2)
        n = self.conv_normal(x)



        return m, n



class Regressor_n(nn.Module):
    def __init__(self):
        super(Regressor_n, self).__init__()
        conv=[
            Transition_block(64, 64),
            Dense_block(64, 32),
            nn.Conv2d(64+32*2, 64, 3, 1, 1),  # 64x4x4
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        ]
        fcn2=[
            # nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 64),
            # nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(64,3)
        ]
        self.fcn2 = nn.Sequential(*fcn2)
        self.conv = nn.Sequential(*conv)
    def forward(self,ln):
        ln=self.conv(ln)
        n=self.fcn2(ln.view(-1,64*4*4))
        n = nn.functional.normalize(n, p=2, dim=1)
        return n

class Decode_shallow(nn.Module):

    def __init__(self):
        super(Decode_shallow, self).__init__()
        dec_m= [
            nn.ConvTranspose2d(64,128, 3, 1, 1),  # 256x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2, 0),  # 256x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),  # 256x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ]
        dec_n= [
            nn.ConvTranspose2d(64, 16 * 2 + 64, 3, 1, 1),  # 256x16x16
            # nn.InstanceNorm2d(16 * 2 + 64),
            nn.ReLU(),
            Transition_blockT(16 * 2 + 64, 32 * 2 + 64),
            Dense_blockT(32, 64),
        ]

        dec_all=[
            Transition_blockT(128, 32 * 2 + 32),
            Dense_blockT(32, 32, bn=False),
            # nn.ConvTranspose2d(32, 64, 3, 1, 1),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 1, 1, 0),  # 256x16x16
            nn.Tanh(),
        ]
        self.dec_m = nn.Sequential(*dec_m)
        self.dec_n = nn.Sequential(*dec_n)
        self.dec_all = nn.Sequential(*dec_all)

    def forward(self, m,n):
        m=self.dec_m(m.view(-1, 64, 8, 8))
        n=self.dec_n(n.view(-1, 64, 8, 8))
        input = torch.cat((n,m), dim=1)
        x = self.dec_all(input)
        return x

class CNNPS_net(nn.Module):
    def __init__(self):
        super(CNNPS_net, self).__init__()
        a=1
        self.scale=a
        conv_layers1 = [
            nn.Conv2d(3, 16*a, 3, 1, 1),
            Dense_block(16*a, 16*a),
            Transition_block(16 * 2 * a + 16 * a, 48 * a),
            Dense_block(48 * a, 16 * a),
            nn.Conv2d(16 * 2 * a + 48 * a, 16 * a, 1, 1, 0),  # 16x16x16
            Dense_block(16 * a, 16 * a),
            Transition_block(16 * 2 * a + 16 * a, 48 * a),# 48x8x8
            Dense_block(48 * a, 16 * a),
            nn.Conv2d(16 * 2 * a + 48 * a, 80 * a, 1, 1, 0),  # 80x8x8
        ]
        fc=[
            nn.Linear(80*8*8*a,128*a),
            nn.ReLU(),
            nn.Linear(128*a,3),
        ]
        self.conv_layers1 = nn.Sequential(*conv_layers1)
        self.fc = nn.Sequential(*fc)
    def forward(self, input):
        a=self.scale
        x = self.conv_layers1(input)
        x = self.fc(x.view(-1, 80*8*8*a))
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

