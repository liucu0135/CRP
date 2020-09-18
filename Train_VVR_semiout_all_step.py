import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd


from Q_net_VVR_conv import Q_net_VVR
# from Q_net_VVR_semiOUT import Q_net_VVR
from VVR_Simulator import VVR_Simulator as Simulator
from Replay_Buffer_VVR import myBuffer


# @ single cc:test1: nc 180 cc 288  test2:  nc 180 cc 292
# @ original nc performance:
# @ batching nc: test1: nc 145 cc 1000 test2: nc 165 cc 1000
# @ single nc:test1: nc 153 cc 1200  test2:  nc 165 cc 988


#  @ single target on distortion: test 1:   nc: 255, cc: 65     test2: nc: 255 cc:65
#  @ single target on color change: test 1: nc: 90,  cc: 400    test2: nc 200 cc 680
#  @ scalarized two targets:        test 1: nc: 92,  cc: 210



noc = [10, 20, 10, 20, 10, 20]  # number of colors
# noc = [14, 14, 14, 14, 14, 14]  # number of colors
# nom = [7, 10, 7, 10, 7, 10]  # number of models
nom = [5, 10, 5, 10, 5, 10]  # number of models
ll = [6, 6, 8, 8, 10, 10]  # length of lane
nol = [5, 5, 7, 7, 10, 10]  # number of lanes
nof = [7, 7, 6, 6, 6, 6]  # percentage of filling level(nof*10%)
data = [noc, nom, nol, ll, nof]  # parameters
columns = ['colors', 'models', 'lanes', 'lane length', 'filling', 'color change']
ccs = []  # stores number of color changes
training_curves = []  # stores curves for each set of parameters
training_curves_nc = []  # stores curves for each set of parameters

for scale in range(1,10):
    scale_p=scale/10
    scale_n=1-scale_p

    # scale_p=0.5
    # scale_n=scale_p

    set=0
    cc_curve = []
    nc_curve = []
    training_curves.append(cc_curve)
    training_curves_nc.append(nc_curve)

    # setting parameters
    GAMMA = 1  # reward does not decay
    NUM_LANES = nol[set]
    NUM_COLORS = noc[set]
    NUM_MODELS = nom[set]
    LANE_LENGTH = ll[set]
    CAPACITY = NUM_LANES * LANE_LENGTH * nof[set] // 10
    # print("color-models:{}-{}, bank: {}X{}".format(noc[set], nom[set], nol[set], ll[set]))
    print("scale:{}".format(scale))
    train_epi = 50
    lr = 0.001
    torch.cuda.set_device(0)


    sim = Simulator(num_color=NUM_COLORS, num_model=NUM_MODELS, capacity=CAPACITY, num_lanes=NUM_LANES,
                    lane_length=LANE_LENGTH,cc_file='./csv_files/cost.csv')    #orders are sampled from uniform distribution
                    # lane_length=LANE_LENGTH, color_dist_file='./csv_files/total_orders.csv') # orders are sampled from a specific distribution
                    # lane_length=LANE_LENGTH,cc_file='./csv_files/cost.csv', color_dist_file='./csv_files/total_orders.csv') # orders are sampled from a specific distribution

    policy_net = Q_net_VVR(c=NUM_COLORS, m=NUM_MODELS, num_lane=NUM_LANES, lane_length=LANE_LENGTH, ccm=sim.ccm, include_last=True, pre=scale_p).cuda()
    target_net = Q_net_VVR(c=NUM_COLORS, m=NUM_MODELS, num_lane=NUM_LANES, lane_length=LANE_LENGTH, ccm=sim.ccm, include_last=True, pre=scale_p).cuda()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = StepLR(optimizer,1,0.85)
    buff = myBuffer()  # replay buffer
    eps = 0.9
    acc = []
    lcc = []
    maxr = 0
    avgq = 0
    sync_inter = 5  # synchronization interval of the policy net and target net (/episode)
    max_performance = 10000 #initialize max cc
    bench_flag = False
    mean_cc = np.zeros(5)

    for episode in range(train_epi):
        cumulated_reward = 0
        closs = []
        end = False
        sim.reset()
        steps = 0
        cc_count = 0
        nc_count = 0

        while (not end):
            if eps > 0.0005:
                eps = eps * 0.9995

            policy_net.eval()
            mask = sim.get_action_out_mask()  # get invalid colors
            state = sim.get_out_tensor(gpu=True)  # get states
            last_color = sim.last_color
            a = policy_net.select_action(state, mask, eps=eps)
            # a = sim.rule_select_color()  #for benchmarking
            sim.set_target(a)  # setting the target color in env
            reward=0
            if not a==last_color:
                nc_count += 1
                reward = -1*scale_n

            if not sim.search_color():
                print('wrong color selected')
            # #  processing cars in the same color as a batch
            # while (sim.search_color() or steps == 0):
            result = sim.step_forward_out_semi_rl()

            if result < 0:# for debug
                print('wrong!')
            cc_count+=sim.get_distortion(absolute=True, tollerance=15)/5
            # if np.isnan(distortion).any():
            #     print('nan found in network')
            # reward +=-sim.get_distortion(absolute=True, tollerance=15)/5*scale_p
            reward +=(1-sim.get_distortion()/100)*scale_p/5
            steps += 1
            if steps >= 300 - CAPACITY:
                end = True


            cumulated_reward += reward
            maskp = sim.get_action_out_mask()
            state_p = sim.get_out_tensor()
            buff.record(state=[state[0].cpu(), state[1].cpu()], action=a, state_p=state_p, end=end, reward=reward,
                        mask=mask, maskp=maskp)

            # Train one frame/ update network once
            policy_net.train()
            if (len(buff.sm) > 800):
                s, a, r, sp, m, mp, e = buff.get_sample_tensor(128)
                state_action_values = policy_net(s).gather(1, a)  # prediction Q
                next_state_valuesp = policy_net(sp).detach() - 10000000 * mp  # applying the mask
                next_state_valuest = target_net(sp).detach()  # double dqn trick
                acts_to_take = next_state_valuesp.max(1)[1].unsqueeze(1)  # this action is selected by the agent(policy)
                next_state_values = next_state_valuest.gather(1, acts_to_take)  # this value is estimated by the benchmark network(target)

                # if the episode ends the Q value would be 0 since there will not be any reward to expect
                expected_state_action_values = r + (next_state_values)*(1 - e)

                loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                closs.append(loss.detach().cpu())

        # synchronize the target network and policy network
        if episode % sync_inter == 1:
            # target_net.load_state_dict(policy_net.state_dict())
            # # optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr)
            # optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
            scheduler.step()

        #  record the required numbers
        if len(closs):
            lcc.append(np.mean(closs))
        acc.append(cumulated_reward)
        cc_curve.append(cc_count)
        nc_curve.append(nc_count)
        # mean_cc[episode % 5] = nc_count
        mean_cc[episode % 5] = cc_count
        if cumulated_reward > maxr:
            maxr = cumulated_reward

        if episode>100:
            sync_inter=4

        if len(lcc)>0:
            print("episode:{} done, loss:{}, total_reward:{}, eps:{}, ave_q:{}, nc: {}, cc:{}, mean_cc:{}".format(
                episode, np.sqrt(np.mean(lcc)), cumulated_reward, eps,
                torch.sqrt(torch.mean(expected_state_action_values ** 2)), nc_count,
                cc_count, np.mean(mean_cc)))

        if episode > 20:
            if max_performance > np.mean(mean_cc):
                policy_net.save_with_num('saved_unshaped', set)
                max_performance = np.mean(mean_cc)
                print("model saved")

    ccs.append(max_performance)  # record performance after each episode
data.append(ccs)
result = pd.DataFrame.from_records(data)
curve = pd.DataFrame.from_records(training_curves)
curve_nc = pd.DataFrame.from_records(training_curves_nc)
result.to_csv('./results/results2.csv')
curve.to_csv('./results/curves2.csv')
curve_nc.to_csv('./results/curves_nc2.csv')
