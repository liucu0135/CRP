from VVR_Bank import VVR_Bank as Bank
import numpy as np
import torch


class VVR_Simulator():
    def __init__(self, num_color=6, num_model=3, capacity=30, num_lanes=6, lane_length=7):
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.target_color=-1
        self.lane_length = lane_length
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.start_sequencec = np.random.randint(0, self.num_color, 1500).tolist()
        self.start_sequencem = np.random.randint(0, self.num_model, 1500).tolist()
        self.bank = Bank(fix_init=True, num_of_colors=self.num_model, sequence=self.start_sequencem,
                         num_of_lanes=self.num_lanes,
                         lane_length=self.lane_length)
        self.mc_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.last_color = -1
        self.job_list = np.zeros(self.num_model)
        self.cc = -1
        for i in range(self.capacity):
            self.BBA_rule_step_in()

    def get_tensor(self, gpu=False):
        in_tensor = torch.zeros(self.num_model)
        last_tensor = torch.zeros(self.num_color)
        steps = torch.ones(self.num_color).float()*len(self.bank.out_queue)/1000
        if self.last_color>-1:
            last_tensor[self.last_color]=1
        if len(self.bank.in_queue):
            in_m=self.bank.in_queue[0]
            in_tensor[in_m]=1
            in_tensor=in_tensor.unsqueeze(1).unsqueeze(1)
            in_tensor=in_tensor.repeat(1,self.num_lanes,1)
            in_tensor=torch.cat((torch.FloatTensor(self.bank.state),in_tensor), dim=2)
        mct=torch.FloatTensor(self.mc_tab)
        mct=torch.cat((mct,last_tensor.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,steps.unsqueeze(0)),dim=0)

        if gpu:
            return in_tensor.cuda().float(), mct.cuda()
        else:
            return in_tensor.float(), mct

    def get_out_tensor(self, gpu=False):
        in_tensor = torch.zeros(self.num_model)
        last_tensor = torch.zeros(self.num_color)
        m_hist = torch.Tensor(self.bank.front_hist()).float()
        m_hist2 = torch.Tensor(self.bank.front_hist(scope=2)).float()
        m_hist = torch.cat([m_hist, torch.zeros(self.num_color-self.num_model)])
        m_hist2 = torch.cat([m_hist2, torch.zeros(self.num_color-self.num_model)])
        steps = torch.ones(self.num_color).float()*len(self.bank.out_queue)/100
        if self.last_color>-1:
            last_tensor[self.last_color]=1
        if len(self.bank.in_queue):
            in_m=self.bank.in_queue[0]
            in_tensor[in_m]=1
            in_tensor=in_tensor.unsqueeze(1).unsqueeze(1)
            in_tensor=in_tensor.repeat(1,self.num_lanes,1)
            in_tensor=torch.cat((torch.FloatTensor(self.bank.state),in_tensor), dim=2)
        mct=torch.FloatTensor(self.mc_tab)
        mct=torch.cat((mct,m_hist2.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,steps.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist.unsqueeze(0)),dim=0)

        if gpu:
            return in_tensor.cuda().float(), mct.cuda()
        else:
            return in_tensor.float(), mct

    def VVR_rule_out(self):
        if self.last_color==-1:
            last=self.target_color
        else:
            last=self.last_color
        if sum(self.job_list) == 0:
            self.job_list, last_color = self.select_color(last)
        else:
            last_color = self.last_color
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (self.job_list[model]) and (model > -1) > 0:
                return self.release(l, last_color)
        print("empty")

    def set_target(self,c):
        self.last_color=c

    def select_color(self, last_color):
        m_hist = self.bank.front_hist()
        max_jl = np.zeros(self.num_model)
        jl = np.minimum(self.mc_tab[:, last_color], m_hist)
        color = None
        if (sum(jl) > 0):
            return jl, last_color
        else:
            print("not posiible")
            print(self.bank.get_view_state())
            print(self.mc_tab)
            return jl, last_color

    def rule_select_color(self):
        m_hist = self.bank.front_hist()
        max_jl = np.zeros(self.num_model)
        color = None
        for c in range(self.num_color):
            jl = np.minimum(self.mc_tab[:, c], m_hist)
            if sum(jl) > sum(max_jl):
                max_jl = jl.copy()
                color = c
        if color == None:
            print("not posiible")
            print(self.bank.get_view_state())
            print(self.mc_tab)
        return color


    def search_color(self):# try to find if there are any car in "last_color"
        m_hist = self.bank.front_hist()
        last_color=self.last_color
        jl = np.minimum(self.mc_tab[:, last_color], m_hist)
        return (last_color > -1) and (sum(jl) > 0)

    def balance_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        l = np.argmin(self.bank.cursors)
        r = self.bank.insert(l)
        self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r

    def ME_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        fv = self.bank.front_view()
        candidate = np.zeros(self.num_lanes)

        for l in range(self.num_lanes):
            candidate[l] = self.mc_tab[fv[l], color]
            if self.bank.cursors[l] == self.lane_length:
                candidate[l] = -1

        c = candidate  # + (self.lane_length - self.bank.cursors)

        r = self.bank.insert(np.argmax(c))
        self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r



    def random_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        l = np.random.randint(0, self.num_lanes)
        r = self.bank.insert(l)
        self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r

    def action_in(self,a):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        r=self.bank.insert(a)
        self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r

    # def get_action_out_mask(self):
    #     fm=torch.zeros(self.mc_tab.shape)
    #     mask=self.mc_tab>0
    #     mask=torch.tensor(mask.astype(np.float32))
    #     fv=self.bank.front_view()
    #     for v in fv:
    #         if v>-1:
    #             fm[v,:]=fm[v,:]+1
    #     mask=mask*fm
    #     mask=mask.view(-1).float()
    #     return mask

    def get_action_out_mask(self):
        mask=torch.zeros(self.num_color)
        m_hist = self.bank.front_hist()
        for c in range(self.num_color):
            jl = np.minimum(self.mc_tab[:, c], m_hist)
            if sum(jl) > 0:
                mask[c]=1
        return mask

    def action_out(self,a):
        c = a % self.num_color
        m = (a - c) // self.num_color

        fv = self.bank.front_view()
        for l in range(self.num_lanes):
            if fv[l]==m:
                return self.release(l,c)

    def step_forward_out(self, a, bench=False):
        last = self.last_color
        if bench:
            if self.BBA_rule_step_in():
                if self.VVR_rule_out():
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]

        else:
            if self.BBA_rule_step_in():
                if self.action_out(a):
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]

    def step_forward_VVR(self, a, bench=False):
        last = self.last_color
        if bench:
            if self.BBA_rule_step_in():
                if self.VVR_rule_out():
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]

        else:
            if self.action_in(a):
                if self.VVR_rule_out():
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]

    def step_forward_out_semi_rl(self):
        last = self.last_color
        if 1:
            if self.BBA_rule_step_in():
                if self.VVR_rule_out():
                    if last==-1:
                        return self.rewards[0]
                    else:
                        if last==self.last_color:
                            return self.rewards[0]
                        else:
                            print('wrong!')
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]






    def BBA_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        c = self.bank.check_rear(model)
        if max(c) > 0:
            r = self.bank.insert(np.argmax(c))
            self.mc_tab[model, color] += r
        else:
            c = self.bank.check_all(model)
            if max(c) > 0:
                r = self.bank.insert(np.argmax(c))
                self.mc_tab[model, color] += r
            else:
                r = self.bank.insert(np.argmax(-self.bank.cursors))
                self.mc_tab[model, color] += r
        if r:
            self.start_sequencec.pop()
        return r

    def release(self, lane, color):
        model = self.bank.front_view()
        model = model[lane]
        if model < 0:
            return False  # return false if the lane is empty
        if self.mc_tab[model, color] > 0:
            if not self.bank.release(lane):
                return False  # return false if the release fails
            self.mc_tab[model, color] -= 1
            self.job_list[model] -= 1
            if not (color == self.last_color):
                self.cc += 1
                self.last_color = color
            return True  # return True if the release success
        else:
            return False  # return false if the corresponding model and color was not needed anymore
