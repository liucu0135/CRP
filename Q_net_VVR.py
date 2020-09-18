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
            Dense_block(16, 16),
            nn.ReLU(),
            nn.Conv2d(16*3, 64, 3, 1, 1),
            # Dense_block(36, 16),
            nn.ReLU(),
            nn.Conv2d(64, 8, 1, 1, 0)
        ]
        # conv_m = [
        #     nn.Conv2d(m, 32, 3,1,1),
        #     Dense_block(32, 32),
        #     nn.ReLU(),
        #     nn.Conv2d(32+32*2, 72, 1, 1, 0),
        #     Dense_block(72, 32),
        #     nn.ReLU(),
        #     nn.Conv2d(72+32*2, 8, 1, 1, 0)
        # ]
        fc0=[
            nn.Linear(8*self.fcnum, 64),
            nn.ReLU(),
        ]
        fc1=[
            nn.Linear(64+(m + 2) * c, 128),
            nn.ReLU(),
            nn.Linear(128, num_lane),
        ]
        # self.conv_c=nn.Sequential(*conv_c)
        self.conv_m=nn.Sequential(*conv_m)
        self.fc0=nn.Sequential(*fc0)
        self.fc1=nn.Sequential(*fc1)
        self.color_num=c

    def forward(self, s):
        state_m=s[0]
        tab_mc=s[1]
        if not self.training:
            state_m=state_m.unsqueeze(0)
            tab_mc=tab_mc.unsqueeze(0)

        xm=self.conv_m(state_m)
        xm=self.fc0(xm.view(-1,8*self.fcnum))
        x=torch.cat([xm, tab_mc.view(-1, (self.m+2)*self.c)],dim=1)
        x=self.fc1(x)
        return x

    def save(self, path):
        torch.save(self.cpu().state_dict(), path + '/net.path')
        self.cuda()

    def load(self, path):
        self.load_state_dict(torch.load(path + '/net.path'))
        self.cuda()



    def select_action(self, state, mask, eps=0):
        actions = self.forward(state) - 1000000 * mask.cuda()
        if random.uniform(0,1)>eps:
            act = torch.argmax(actions)
            return int(act.detach().cpu().numpy())
        else:
            # actions=random.uniform(0,1,6)-100*mask
            actions=nn.functional.softmax(actions)
            actions=actions.squeeze().detach().cpu().numpy()
            act=random.choice(range(self.num_lane), 1, p=actions)[0]
            return act