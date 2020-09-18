from Bank import Bank
import numpy as np
import torch

class Simulator():
    def __init__(self, num_color=10):
        self.bank=Bank(num_of_colors=num_color)
        self.rewards=[1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity=30
        self.start_sequence=np.random.randint(0,num_color,50).tolist()
        self.num_color=num_color
        # self.reset()

    def reset_rng(self):
        self.start_sequence = np.random.randint(0, self.num_color, 50).tolist()

    def reset(self):
        self.reset_rng()
        self.bank = Bank(fix_init=True, sequence=self.start_sequence.copy(), num_of_colors=self.num_color)
        for i in range(self.capacity):
            self.bank.insert(i%6)

    def step(self, in_num, out_num):
        if not self.bank.insert(in_num): return self.rewards[2]
        if not self.bank.release(out_num): return self.rewards[2]
        if self.bank.check_cc():
            return self.rewards[1]
        else:
            return self.rewards[0]

    def step_out(self, out_num):
        if not self.bank.release(out_num): return self.rewards[2], sum(self.bank.cursors)==0
        if self.bank.check_cc():
            return self.rewards[1], sum(self.bank.cursors)==0
        else:
            return self.rewards[0], sum(self.bank.cursors)==0

    def get_tensor(self, gpu=False):
        out_tensor = torch.zeros(self.bank.num_of_colors)
        if len(self.bank.out_queue):
            out=self.bank.out_queue[-1]
            out_tensor[out]=1
        out_tensor=out_tensor.unsqueeze(1).unsqueeze(1)
        out_tensor=out_tensor.repeat(1,6,1)
        out_tensor=torch.cat((torch.FloatTensor(self.bank.state),out_tensor), dim=2)
        if gpu:
            return out_tensor.cuda().float()
        else:
            return out_tensor.float()

    def get_view_state(self):
        return self.bank.get_view_state()


# # for test
# s= Simulator()
# for i in range(20):
#     print(s.get_view_state())
#     IN=np.random.randint(0,6)
#     OUT=np.random.randint(0,6)
#     print(OUT, s.step_out(OUT))
