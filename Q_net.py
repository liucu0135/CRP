import torch
import torch.nn as nn
import numpy.random as random
from Component import Dense_block

class Q_net_out(nn.Module):
    def __init__(self, c):
        super(Q_net_out, self).__init__()
        conv = [
            nn.Conv2d(c, 16, 3,1,1),
            Dense_block(16, 16),
            nn.ReLU(),
            nn.Conv2d(16+16*2, 36, 3, 1, 1),
            Dense_block(36, 16),
            nn.ReLU(),
            nn.Conv2d(36+16*2, 16, 3, 1, 1),
            # # nn.ReLU(),
            # nn.Conv2d(64, 128, 1, 1, 0),
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 1,1,0),
            # nn.ReLU(),
        ]
        fc=[
            nn.Linear(16*6*9, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        ]
        self.conv=nn.Sequential(*conv)
        self.fc=nn.Sequential(*fc)
        self.color_num=c


    def save(self, path):
        torch.save(self.cpu().state_dict(), path + '/net.path')
        self.cuda()

    def load(self, path):
        self.load_state_dict(torch.load(path + '/net.path'))
        self.cuda()

    def forward(self, state):
        if not self.training:
            bank=state.unsqueeze(0)
        else:
            bank=state

        x=self.conv(bank)
        # x=torch.cat([x.view(-1, 32*6*8),out],dim=1)
        x=self.fc(x.view(-1, 16*6*9))
        return x

    def select_action(self, state, mask, eps=0):
        actions = self.forward(state) - 100 * mask.cuda()
        if random.uniform(0,1)>eps:
            act = torch.argmax(actions)
            return int(act.detach().cpu().numpy())
        else:
            # actions=random.uniform(0,1,6)-100*mask
            actions=nn.functional.softmax(actions)
            actions=actions.squeeze().detach().cpu().numpy()
            act=random.choice(range(6), 1, p=actions)[0]
            return act