import torch
import torch.nn as nn
import numpy.random as random
import numpy as np
from Component import Dense_block

class Q_net_VVR(nn.Module):
    def __init__(self, c, m, num_lane=7, lane_length=8, ccm=None, include_last=False):
        super(Q_net_VVR, self).__init__()

        #  setting some parameters
        if (ccm is None)or(include_last):
            self.ccm = torch.ones(c, c).cuda()
        else:
            self.ccm = torch.Tensor(ccm).transpose(1, 0).cuda()
        self.include_last=include_last  # if last color is taken as state
        self.m=m
        self.c=c
        self.fcnum=(num_lane*(lane_length+2))
        self.num_lane=num_lane

        # setting a couple of layers
        conv0=[
            nn.Conv2d(self.m, 8, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        ]

        conv1=[
            nn.Conv2d(4+self.include_last*2, 8, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        ]

        fc0 = [
            nn.Linear(32*self.fcnum, 256),
            nn.ReLU(),
        ]
        fc1=[
            nn.Linear(32 * self.c*self.m, 256),
            nn.ReLU(),
        ]
        fc_fuse=[
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ]
        self.fca=nn.Linear(128, c)
        # self.fca=nn.Linear(128+self.c, c)
        self.fcv=nn.Linear(128+1, 1)
        self.conv0=nn.Sequential(*conv0)
        self.conv1=nn.Sequential(*conv1)
        self.fc1=nn.Sequential(*fc1)
        self.fc0=nn.Sequential(*fc0)
        self.fc_fuse=nn.Sequential(*fc_fuse)
        self.color_num=c


    def forward(self, s):
        state=s[0]
        tab_mc=s[1]
        # if not training means the batch size doesn't exist, unsqueezing the data to match dimension
        if not self.training:
            state=state.unsqueeze(0)
            tab = tab_mc[:self.m, :self.c].unsqueeze(0).unsqueeze(0)
            if self.include_last:
                dist_alert=tab_mc[-6, :self.c].repeat(self.m, 1).unsqueeze(0).unsqueeze(0)
                last = tab_mc[-5, :self.c].repeat(self.m, 1).unsqueeze(0).unsqueeze(0)
            else:
                last = tab_mc[-5, :self.c].repeat(self.m, 1).mm(self.ccm).unsqueeze(0).unsqueeze(0)

            step = tab_mc[-4, 0].unsqueeze(0).unsqueeze(0)
            hist = tab_mc[-3, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3,2)
            hist2 = tab_mc[-2, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3,2)
            hist3 = tab_mc[-1, :self.m].repeat(1, self.c, 1).unsqueeze(0).transpose(3,2)
        else:
            tab = tab_mc[:, :self.m, :self.c].unsqueeze(1)
            if self.include_last:
                dist_alert = tab_mc[:, -6, :self.c].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
                last = tab_mc[:, -5, :self.c].unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
            else:
                last = tab_mc[:, -5, :self.c].mm(self.ccm).unsqueeze(1).repeat(1, self.m, 1).unsqueeze(1)
            step = tab_mc[:, -4, 0].unsqueeze(1)
            hist = tab_mc[:, -3, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3,2)
            hist2 = tab_mc[:, -2, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3,2)
            hist3 = tab_mc[:, -1, :self.m].unsqueeze(1).repeat(1, self.c, 1).unsqueeze(1).transpose(3,2)

        if self.ccm is None:
            hist = torch.min(tab, hist) * last
            hist2 = torch.min(tab, hist2) * last
            hist3 = torch.min(tab, hist3) * last
            x = torch.cat((tab, hist, hist2, hist3), dim=1)
        elif self.include_last:
            hist = torch.min(tab, hist)
            hist2 = torch.min(tab, hist2)
            hist3 = torch.min(tab, hist3)
            x = torch.cat((tab, hist, hist2, hist3, last, dist_alert), dim=1)
        else:
            hist = torch.min(tab, hist) * last
            hist2 = torch.min(tab, hist2) * last
            hist3 = torch.min(tab, hist3) * last
            x = torch.cat((tab, hist, hist2, hist3), dim=1)



        x1=self.conv1(x)
        x1=self.fc1(x1.view(-1,32 * self.c* self.m))
        x2 = self.conv0(state)
        x2 = self.fc0(x2.view(-1, 32 * self.fcnum))
        x=self.fc_fuse(torch.cat((x1,x2), dim=1))
        v=self.fcv(torch.cat((x[:,:128],step),dim=1))
        a=self.fca(x[:,128:])
        # a=self.fca(torch.cat((x[:,128:], dist_alert[:,0,0,:]), dim=1))
        m=torch.mean(a,dim=1).unsqueeze(1).repeat(1,a.shape[1])
        a=a-m
        x=a+v
        return x

    def save(self, path):
        torch.save(self.cpu().state_dict(), path + '/net.path')
        self.cuda()

    def save_with_num(self, path,num):
        torch.save(self.cpu().state_dict(), path + '/net{}.path'.format(num))
        self.cuda()

    def load(self, path, num=None):
        if num is None:
            self.load_state_dict(torch.load(path + '/net.path'))
        else:
            self.load_state_dict(torch.load(path + '/net{}.path'.format(num)))
        self.cuda()

    def load_with_num(self, path, num):
        self.load_state_dict(torch.load(path + '/net{}.path'.format(num)))
        self.cuda()


    def select_action(self, state, mask, eps=0):
        q_values=self.forward(state)

        actions = q_values - 2*torch.max(torch.abs(q_values)) * (1-mask).cuda()
        # print('output:',actions.cpu().detach().numpy()//1)
        # actions = actions.view(self.m, self.c)
        if random.uniform(0,1)>eps:
            act = torch.argmax(actions)
            act= int(act.detach().cpu().numpy())
        else:
            actions=nn.functional.softmax(actions, dim=1)
            actions=actions.squeeze().detach().cpu()*(mask)
            actions=actions.numpy()/np.sum(actions.numpy())
            # print(sum(actions))
            act=random.choice(range(self.c), 1, p=actions)[0]
        return act