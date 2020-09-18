from VVR_Bank import VVR_Bank as Bank
from VVR_Simulator import VVR_Simulator as Sim

preFill=21
s = Sim(num_color=20, num_model=10, num_lanes=5, lane_length=6, capacity=preFill)
print(s.mc_tab)
print(s.bank.get_view_state())
print(s.bank.front_hist())
print(s.bank.front_view())

cc_sum=0
for _ in range(10):
    s.reset()
    for i in range(1000):
        if s.VVR_rule_out():
            # print("released:")
            # print(s.mc_tab)
            # print(s.bank.get_view_state())
            # print("joblist",s.job_list)
            # print("color:", s.last_color)
            if i<1000-preFill:
                while not s.ME_rule_step_in(): _
        else:
            print("release failed")
    cc_sum+=s.cc
print(cc_sum/10)
