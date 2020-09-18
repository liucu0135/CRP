import torch
import torch.nn as nn
import numpy.random as random
from Component import Dense_block

class Q_net_VVR(nn.Module):
    def __init__(self, c, m, num_lane=7, lane_length=8):
        super(Q_net_VVR, self).__init__()
        self.m=m
        self.c=c
        self.fcnum=(num_lane*(lane_length+2))
        self.num_lane=num_lane

        conv_m = [
            nn.Conv2d(m, 16, 3,1,1),
            Dense_block(16, 8),
            nn.ReLU(),
            nn.Conv2d(8*2+16, 8, 3, 1, 1),
            # Dense_block(36, 16),
            nn.ReLU(),
            nn.Conv2d(8, 8, 1, 1, 0)
        ]
        fc0=[
            nn.Linear(8*self.fcnum, 8),
            nn.ReLU(),
        ]
        conv0=[
            nn.Conv2d(3, 128, [self.m,1]),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(),
        ]
        fc1=[
            nn.Linear(128 * self.c, 256),
            nn.ReLU(),
        ]
        self.fca=nn.Linear(128, m*c)
        self.fcv=nn.Linear(128+1, 1)
        self.conv0=nn.Sequential(*conv0)
        # self.conv_m=nn.Sequential(*conv_m)
        # self.fc0=nn.Sequential(*fc0)
        self.fc1=nn.Sequential(*fc1)
        self.color_num=c

    def forward(self, s):
        state_m=s[0]
        tab_mc=s[1]
        if not self.training:
            tab = tab_mc[:-3, :].unsqueeze(0).unsqueeze(0)
            last = tab_mc[-3, :].repeat(1, self.m, 1).unsqueeze(0)
            step = tab_mc[-2, 1].unsqueeze(0).unsqueeze(0)
            hist = tab_mc[-1, :].repeat(1, self.m, 1).unsqueeze(0)
        else:
            tab = tab_mc[:, :-3, :].unsqueeze(1)
            last = tab_mc[:, -3, :].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
            step = tab_mc[:, -2, 1].unsqueeze(1)
            hist = tab_mc[:, -1, :].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)

        # xm=self.conv_m(state_m)
        # xm=self.fc0(xm.view(-1,8*self.fcnum))
        # x=torch.cat([xm, tab_mc.view(-1, (self.m+3)*self.c)],dim=1)

        x=torch.cat((tab,last,hist),dim=1)
        x=self.conv0(x)



        x=x.view(-1,128 * self.c)


        x=self.fc1(x)
        v=self.fcv(torch.cat((x[:,:128],step),dim=1))
        a=self.fca(x[:,128:])
        m=torch.mean(a,dim=1).unsqueeze(1).repeat(1,a.shape[1])
        a=a-m
        x=a+v
        return x

    def save(self, path):
        torch.save(self.cpu().state_dict(), path + '/net.path')
        self.cuda()

    def load(self, path):
        self.load_state_dict(torch.load(path + '/net.path'))
        self.cuda()



    def select_action(self, state, mask, eps=0):
        actions = self.forward(state) - 1000000 * (1-mask).cuda()
        # actions = actions.view(self.m, self.c)
        if random.uniform(0,1)>eps:
            act = torch.argmax(actions)
            act= int(act.detach().cpu().numpy())
        else:
            # actions=random.uniform(0,1,6)-100*mask
            actions=nn.functional.softmax(actions)
            actions=actions.squeeze().detach().cpu().numpy()
            act=random.choice(range(self.m*self.c), 1, p=actions)[0]
        return act