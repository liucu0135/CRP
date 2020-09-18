import torch

from Q_net_VVR_semiOUT import Q_net_VVR
from VVR_Simulator import VVR_Simulator as Simulator
from Replay_Buffer_VVR import myBuffer
import numpy as np
import pandas as pd
import time

noc = [14, 14, 14, 14, 14, 14]
nom = [7, 10, 7, 10, 7, 10]
ll = [6, 6, 8, 8, 10, 10]
nol = [5, 5, 7, 7, 10, 10]
nof = [7, 7, 6, 6, 6, 6]
ccs = []
sets=[]
cc = []
ts = []
times=[]
data = [noc, nom, nol, ll, nof]
columns = ['colors', 'models', 'lanes', 'lane length', 'filling', 'color change','time']
training_curves = []

for set in range(6):
    cc_curve = []
    training_curves.append(cc_curve)
    GAMMA = 1
    NUM_LANES = nol[set]
    NUM_COLORS = noc[set]
    NUM_MODELS = nom[set]
    LANE_LENGTH = ll[set]
    CAPACITY = NUM_LANES * LANE_LENGTH * nof[set] // 10

    train_epi = 10
    torch.cuda.set_device(0)
    lr = 0.00001
    sim = Simulator(num_color=NUM_COLORS, num_model=NUM_MODELS, capacity=CAPACITY, num_lanes=NUM_LANES,
                    lane_length=LANE_LENGTH, cc_file='./csv_files/cost.csv',
                    color_dist_file='./csv_files/total_orders.csv')

    policy_net = Q_net_VVR(c=NUM_COLORS, m=NUM_MODELS, num_lane=NUM_LANES, lane_length=LANE_LENGTH, ccm=sim.ccm).cuda()
    policy_net.load_with_num('checkpoints2',set)
    buff = myBuffer()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    eps = 0
    max_performance = 1000
    bench_flag = False
    mean_cc = np.zeros(train_epi)
    resume = False

    for episode in range(train_epi):
        cumulated_reward = 0
        end = False
        sim.reset()
        steps = 0
        sbuff = []
        rbuff = []
        spbuff = []
        start = True
        cc_count = 0
        cc_sum = 0
        t_count = 0
        start_time=time.time()
        while (not end):
            if steps >= 1000 - CAPACITY:
                end = True
            policy_net.eval()
            mask = sim.get_action_out_mask()
            state = sim.get_out_tensor(gpu=True)
            last_color = sim.last_color
            a = policy_net.select_action(state, mask, eps=eps)
            # a = sim.rule_select_color()
            sim.set_target(a)

            cc_count += sim.ccm[a, last_color]

            while (sim.search_color() or steps == 0):
                rr = sim.step_forward_out_semi_rl()
                if rr < 1:
                    print('wrong!')
                steps += 1
                if steps >= 1000 - CAPACITY:
                    end = True
                    break

            start = False
        times.append(time.time()-start_time)
        t_count+=time.time()-start_time
        cc_sum+=cc_count
        cc.append(cc_count)
        sets.append(set)
        mean_cc[episode] = cc_count

        print("color-models:{}-{}, bank: {}X{},epi:{}, cc:{}".format(noc[set], nom[set], nol[set], ll[set],episode, cc_count))
        if episode == train_epi-1:
            print("color-models:{}-{}, bank: {}X{}, cc:{}".format(noc[set], nom[set], nol[set], ll[set], np.mean(mean_cc)))
    ccs.append(np.mean(mean_cc))
data.append(ccs)
data.append(ts)
data.append(cc)
data.append(times)
result = pd.DataFrame.from_records(data)
result.to_csv('./results/results_ccmat.csv')