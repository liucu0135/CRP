import torch.utils.data as Data
import torch
import numpy.random as random
import numpy as np


class myBuffer():
    def __init__(self, buffersize=10000):
        self.sm=[]
        self.smc=[]
        self.smp=[]
        self.smcp=[]
        self.a=[]
        self.r=[]
        self.e=[]
        self.m=[]
        self.mp=[]
        self.index=0

    def __len__(self):
        return len(self.sm)

    def sample_batch(self, batch_size, s, idx):
        if len(s)>batch_size:
            # idx=random.permutation(np.arange(len(s)))[:batch_size]
            ss=[]
            for i in idx:
                ss.append(s[i])
            s=torch.stack(ss)
        else:
            s=torch.stack(s)
        return s

    def sample_int_batch(self, batch_size, s, idx):
        if len(s)>batch_size:
            ss=[]
            for i in idx:
                ss.append(s[i])
            s=torch.Tensor(ss).unsqueeze(1)
        else:
            s=torch.Tensor(s).unsqueeze(1)
        return s

    def sample_state_batch(self, batch_size, s, idx):

        if len(s)<batch_size:
            print("buffer is smaller than batchsize!!!")
            return False
        else:
            ss=[s[i] for i in idx]
            s=torch.stack(ss, dim=0)


        return s

    def get_sample_tensor(self, batch_size):
        idx=random.permutation(np.arange(len(self.sm)))[:batch_size]
        sm=self.sample_state_batch(batch_size, self.sm,idx)
        smc=self.sample_state_batch(batch_size, self.smc,idx)
        smp=self.sample_state_batch(batch_size, self.smp,idx)
        smcp=self.sample_state_batch(batch_size, self.smcp,idx)
        mask=self.sample_batch(batch_size, self.m,idx)
        maskp=self.sample_batch(batch_size, self.mp,idx)
        a=self.sample_int_batch(batch_size, self.a,idx)
        r=self.sample_int_batch(batch_size, self.r,idx)
        e=self.sample_int_batch(batch_size, self.e, idx)
        return [sm.cuda(), smc.cuda()], a.long().cuda(), r.float().cuda(), [smp.cuda(),smcp.cuda()], mask.float().cuda(), maskp.float().cuda(), e.float().cuda()


    def record(self, state, action, reward, state_p,  mask, maskp, end=False):
        if len(self)<10000:
            self.sm.append(state[0])
            self.smc.append(state[1])
            self.a.append(action)
            self.r.append(reward)
            self.smp.append(state_p[0])
            self.smcp.append(state_p[1])
            self.e.append(end)
            self.m.append(mask)
            self.mp.append(maskp)
        else:
            self.sm[self.index]=state[0]
            self.smc[self.index]=state[1]
            self.a[self.index]=action
            self.r[self.index]=reward
            self.smp[self.index]=state_p[0]
            self.smcp[self.index]=state_p[1]
            self.e[self.index]=end
            self.m[self.index]=mask
            self.mp[self.index]=maskp
            if self.index==10000-1:
                self.index=0
            else:
                self.index+=1


