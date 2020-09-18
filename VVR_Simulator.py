from VVR_Bank import VVR_Bank as Bank
import numpy as np
import torch
import pandas as pd


class VVR_Simulator():
    def __init__(self, num_color=6, num_model=3, capacity=30, num_lanes=6, lane_length=7, cc_file=None, color_dist_file=None, vari=False):
        self.num_color = num_color
        self.num_model = num_model
        self.num_lanes = num_lanes
        self.cc_file=cc_file
        self.target_color=-1
        self.lane_length = lane_length
        self.rewards = [1, 0, -1]  # [0]for unchange, [1]for change, [2]for error
        self.capacity = capacity
        self.color_dist_file=None
        if cc_file is not None:
            self.ccm=self.read_cc_matrix(cc_file)
        if color_dist_file is not None:
            self.color_dist=self.read_color_dist(color_dist_file)
            self.model_dist=self.color_dist[:self.num_model]/sum(self.color_dist[:self.num_model])
        self.reset()

    def cc_vari(self,c):
        self.ccm*=c

    def read_color_dist(self,cc_file):
        ccm=pd.read_csv(cc_file)
        ccm=ccm.to_numpy(dtype=np.float,copy=True).squeeze()
        ccm=ccm/np.sum(ccm)
        return ccm

    def read_cc_matrix(self,cc_file):
        ccm=pd.read_csv(cc_file)
        ccm=ccm.to_numpy(dtype=np.float,copy=True)[:,1:]
        return ccm

    def reset(self):
        if self.color_dist_file is not None:
            self.start_sequencec = np.random.choice(range(self.num_color), 1100, p=self.color_dist).tolist()
        else:
            self.start_sequencec = np.random.choice(range(self.num_color), 1100).tolist()
        if self.color_dist_file is not None:
            self.start_sequencem = np.random.choice(range(self.num_model), 1100, p=self.model_dist).tolist()
        else:
            self.start_sequencem = np.random.choice(range(self.num_model), 1100).tolist()
        self.bank = Bank(fix_init=True, num_of_colors=self.num_model, sequence=self.start_sequencem,
                         num_of_lanes=self.num_lanes,
                         lane_length=self.lane_length)
        self.plan_list = {k: [c, m] for k, c, m in
                          zip(range(len(self.start_sequencec)), self.start_sequencec, self.start_sequencem)}
        self.mc_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.dist_tab = np.zeros((self.num_model, self.num_color), dtype=np.int)
        self.last_color = -1
        self.job_list = np.zeros(self.num_model)
        self.cc = -1
        for i in range(self.capacity):
            self.BBA_rule_step_in()


    def get_out_tensor(self, gpu=False, cc=False):
        num_tensor=max(self.num_color,self.num_model)
        in_tensor = torch.zeros(self.num_model)
        mct=torch.zeros(num_tensor,num_tensor)
        last_tensor = torch.zeros(num_tensor)
        dist_alert_tensor = torch.zeros(num_tensor).float()
        ctemp = np.zeros(self.num_color)

        m_hist= torch.zeros(num_tensor)
        m_hist2= torch.zeros(num_tensor)
        m_hist3= torch.zeros(num_tensor)
        m_hist[:self.num_model] = torch.Tensor(self.bank.front_hist()).float()
        m_hist2[:self.num_model] = torch.Tensor(self.bank.front_hist(scope=2)).float()
        m_hist3[:self.num_model] = torch.Tensor(self.bank.front_hist(scope=3)).float()
        steps = torch.ones(num_tensor).float()*len(self.bank.out_queue)/100
        if self.last_color>-1:
            last_tensor[self.last_color]=1
        for c in range(self.num_color):
            ctemp[c]=self.search_model(c)
        for i in range(len(self.start_sequencec)+self.capacity, 1100):
            if i in self.plan_list:
                if ctemp[self.plan_list[i][0]]==self.plan_list[i][1]:
                    dist_alert_tensor[self.plan_list[i][0]]=i-len(self.start_sequencec)+self.capacity
                    continue

        if len(self.bank.in_queue):
            in_m=self.bank.in_queue[0]
            in_tensor[in_m]=1
            in_tensor=in_tensor.unsqueeze(1).unsqueeze(1)
            in_tensor=in_tensor.repeat(1,self.num_lanes,1)
            in_tensor=torch.cat((torch.FloatTensor(self.bank.state),in_tensor), dim=2)
        mct[:self.num_model,:self.num_color]=torch.FloatTensor(self.mc_tab)
        # if self.num_color<self.num_model:
        #     mct = torch.cat((mct, torch.zeros(self.num_model,self.num_model-self.num_color)),dim=1)
        mct=torch.cat((mct,dist_alert_tensor.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,last_tensor.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,steps.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist2.unsqueeze(0)),dim=0)
        mct=torch.cat((mct,m_hist3.unsqueeze(0)),dim=0)

        if gpu:
            return in_tensor.cuda().float(), mct.cuda()
        else:
            return in_tensor.float(), mct

    def VVR_rule_out(self):
        if self.last_color==-1:
            last=self.target_color
        else:
            last=self.last_color
        self.job_list, last_color = self.find_job_list(last, alert=True) # select available models

        # select a model
        lane_dists=np.zeros(self.num_lanes)
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (self.job_list[model]) and (model > -1):
                lane_dists[l]=self.find_nearest_order(model, last_color)
        l=np.argmax(lane_dists)
        if np.max(lane_dists)==0:
            print("empty")
        return self.release(l, last)


    def set_target(self,c):
        self.last_color=c

    def find_job_list(self, last_color, alert=False):
        m_hist = self.bank.front_hist()
        jl = np.minimum(self.mc_tab[:, last_color], m_hist)
        if (sum(jl) > 0) or (not alert):
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
            if self.ccm is not None:
                jl=jl.astype(np.float64)/self.ccm[c,self.last_color]
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
                            print('color is changed')
                            return self.rewards[1]
                else:
                    return self.rewards[2]
            else:
                return self.rewards[2]


    def BBA_rule_step_in(self):
        color = self.start_sequencec[-1]
        model = self.bank.in_queue[-1]
        c = self.bank.check_rear(model)
       # cb = self.bank.check_color_block(model)

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


    def find_nearest_order(self, m,c):
        for k in range(1100):
            if 1100-k in self.plan_list:
                if self.plan_list[1100-k][0]==c and self.plan_list[1100-k][1]==m:
                    return 1100-k
        print('nearest order not found')

    def search_model(self, color):
        job_list, color = self.find_job_list(color)
        for l in range(self.num_lanes):
            model = self.bank.front_view()[l]
            if (job_list[model]) and (model > -1):
                return model
        return 0

    def release(self, lane, color):
        model = self.bank.front_view()
        model = model[lane]
        if model < 0:
            return False  # return false if the lane is empty
        if self.mc_tab[model, color] > 0:
            if not self.bank.release(lane):
                return False  # return false if the release fails
            self.mc_tab[model, color] -= 1
            if not (color == self.last_color):
                self.cc += 1
                self.last_color = color
            k=self.find_nearest_order(model,color)
            self.plan_list.pop(k)
            self.released_order_index=k
            return True  # return True if the release success
        else:
            return False  # return false if the corresponding model and color was not needed anymore
    # def
    def get_distortion(self, absolute=False, tollerance=0):
        if absolute:
            n=np.maximum((self.released_order_index-len(self.start_sequencec)-self.capacity)-tollerance, 0)
            return n
        dist_count=0
        for i in range(len(self.start_sequencec)+self.capacity, 1100+1):
            if i in self.plan_list:
                dist_count+=i-len(self.start_sequencec)-self.capacity
        return dist_count