import torch

from Q_net_VVR_semiOUT import Q_net_VVR
from VVR_Simulator import VVR_Simulator as Simulator
from Replay_Buffer_VVR import myBuffer
import numpy as np
import pandas as pd

noc = [10, 20, 10, 20, 10, 20]
nom = [5, 10, 5, 10, 5, 10]

# nom = [10, 20, 10, 20, 10, 20]
# noc = [10, 5, 10, 5, 10, 5]

ll = [6, 6, 8, 8, 10, 10]
nol = [5, 5, 7, 7, 10, 10]
nof = [7, 7, 6, 6, 6, 6]
ccs = []
data = [noc, nom, nol, ll, nof]
columns = ['colors', 'models', 'lanes', 'lane length', 'filling', 'color change']
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
    print("color-models:{}-{}, bank: {}X{}".format(noc[set], nom[set], nol[set], ll[set]))
    train_epi = 200
    torch.cuda.set_device(1)
    policy_net = Q_net_VVR(c=NUM_COLORS, m=NUM_MODELS, num_lane=NUM_LANES, lane_length=LANE_LENGTH).cuda()
    target_net = Q_net_VVR(c=NUM_COLORS, m=NUM_MODELS, num_lane=NUM_LANES, lane_length=LANE_LENGTH).cuda()
    lr = 0.0005
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    sim = Simulator(num_color=NUM_COLORS, num_model=NUM_MODELS, capacity=CAPACITY, num_lanes=NUM_LANES,
                    lane_length=LANE_LENGTH)
    buff = myBuffer()
    eps = 0.9
    acc = []
    lcc = []
    maxr = 0
    avgq = 0
    sync_inter = 10
    max_performance = 1000
    bench_flag = False
    mean_cc = np.zeros(10)
    resume = False
    if resume:
        policy_net.load('checkpoints',set)
        target_net.load('checkpoints',set)
        eps = 0.0005
        sync_inter = 128

    for episode in range(train_epi):
        cumulated_reward = 0
        closs = []
        end = False
        sim.reset()
        steps = 0
        sbuff = []
        rbuff = []
        spbuff = []
        start = True
        cc_count = 0

        while (not end):
            if eps > 0.0005:
                eps = eps * 0.999

            if steps >= 1000 - CAPACITY:
                end = True

            policy_net.eval()
            # mask = (torch.Tensor(sim.bank.cursors) == LANE_LENGTH).float()

            mask = sim.get_action_out_mask()
            state = sim.get_out_tensor(gpu=True)
            a = policy_net.select_action(state, mask, eps=eps)
            # a = sim.rule_select_color()  #for benchmarking
            sim.set_target(a)

            cc_count += 1
            reward = -1

            ca = sim.search_color()

            while (sim.search_color() or steps == 0):
                rr = sim.step_forward_out_semi_rl()
                if rr < 1:
                    print('wrong!')
                reward += 1
                steps += 1
                if steps >= 1000 - CAPACITY:
                    end = True
                    break

            cumulated_reward += reward
            policy_net.train()
            maskp = sim.get_action_out_mask()
            state_p = sim.get_out_tensor()

            buff.record(state=[state[0].cpu(), state[1].cpu()], action=a, state_p=state_p, end=end, reward=reward,
                        mask=mask, maskp=maskp)
            start = False
            # Train one frame
            if (len(buff.sm) > 256) and steps % 1 == 0:
                s, a, r, sp, m, mp, e = buff.get_sample_tensor(128)
                state_action_values = policy_net(s).gather(1, a)
                next_state_valuesp = policy_net(sp).detach() - 10000000 * mp
                next_state_valuest = target_net(sp).detach()
                acts_to_take = next_state_valuesp.max(1)[1].unsqueeze(1)  # this action is selected by the agent(policy)
                next_state_values = next_state_valuest.gather(1,
                                                              acts_to_take)  # this value is estimated by the benchmark network(target)
                expected_state_action_values = r + (next_state_values) * (1 - e)
                loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                # for param in policy_net.parameters():
                #     param.grad.data.clamp_(-10, 10)
                optimizer.step()
                closs.append(loss.detach().cpu())

        if episode % sync_inter == 0:
            target_net.load_state_dict(policy_net.state_dict())
            optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
            # target_net.eval()
        if len(closs):
            lcc.append(np.mean(closs))

        acc.append(cumulated_reward)
        cc_curve.append(cc_count)
        mean_cc[episode % 10] = cc_count
        if cumulated_reward > maxr:
            maxr = cumulated_reward


        # print("episode:{} done,  eps:{}, , cc:{}".format(episode, eps, cc_count, np.mean
        if (len(buff.sm) > 256):
            print("episode:{0} done, loss:{1}, total_reward:{2}, eps:{3}, ave_q:{4}, cc:{5}, mean_cc:{6}".format(
            episode, np.sqrt(np.mean(lcc)), cumulated_reward, eps, torch.mean(expected_state_action_values),
            cc_count, np.mean(mean_cc)))
        if episode > 10:
            if max_performance > np.mean(mean_cc):
                policy_net.save_with_num('checkpoints1', set)
                max_performance = np.mean(mean_cc)
                print("episode:{0} done, loss:{1}, total_reward:{2}, eps:{3}, ave_q:{4}, cc:{5}, mean_cc:{6}".format(
                    episode, np.sqrt(np.mean(lcc)), cumulated_reward, eps, torch.mean(expected_state_action_values),
                    cc_count, np.mean(mean_cc)))
                print("model saved")
data.append(ccs)
result = pd.DataFrame.from_records(data)
curve = pd.DataFrame.from_records(training_curves)
# result.to_csv('./results/results.csv')
# curve.to_csv('./results/curves.csv')
