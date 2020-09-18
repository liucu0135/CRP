import torch
import torch.nn as nn
import torch.nn.functional as F
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
        state_size=5*5*8+16*10
        self.state_size = state_size
        self.action_size = 10
        self.reward_size = 0
        network_scale = 4
        # S x A -> (W -> R^n). =>. S x W -> (A -> R^n)
        self.affine1 = nn.Linear(state_size,
                                 (state_size) * network_scale * 1)  # used to be 16, 32, 64
        self.affine2 = nn.Linear((state_size) * network_scale * 1,
                                 (state_size ) * network_scale * 2)
        self.affine3 = nn.Linear((state_size ) * network_scale * 2,
                                 (state_size ) * network_scale * 4)
        self.affine4 = nn.Linear((state_size ) * network_scale * 4,
                                 (state_size ) * network_scale * 2)
        self.affine5 = nn.Linear((state_size ) * network_scale * 2,
                                 10 )
        self.color_num=c


    def forward(self, s):
        state=s[0]
        tab_mc=s[1]
        if len(s[0].shape)==3:
            s1=s[0].unsqueeze(0)
            s2=s[1].unsqueeze(0)
        else:
            s1=s[0]
            s2=s[1]
        x=torch.cat((s1.view(-1,5*5*8),s2.view(-1,16*10)),dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)
        return q

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