from VVR_Bank import VVR_Bank as Bank
import numpy as np
from VVR_Simulator import VVR_Simulator as Sim

s = Sim(cc_file='./csv_files/cost.csv', color_dist_file='./csv_files/total_orders.csv')

print(s.mc_tab)
# print(s.bankc.get_view_state())
print(s.bankm.get_view_state())
print(s.bankm.front_hist())
print(s.bankm.front_view())

a = s.mc_tab
b = s.bankm.front_hist()

for c in range(6):
    print(np.minimum(a[:, c], b))
