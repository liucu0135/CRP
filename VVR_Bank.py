import numpy as np
import matplotlib as plt


class VVR_Bank():
    def __init__(self, fix_init=False, sequence=None, num_of_colors=6, num_of_lanes=6, lane_length=7):
        self.num_of_colors = num_of_colors
        self.nl = num_of_lanes
        self.ll = lane_length
        c = self.num_of_colors
        self.state = np.zeros((c, num_of_lanes, lane_length + 1))  # 6x7 bank with 10 colors
        self.cursors = np.zeros(num_of_lanes, dtype=np.int)
        if not fix_init:
            self.in_queue = np.random.randint(0, c, 1000).tolist()
        else:
            self.in_queue = sequence
        self.out_queue = []

    def get_view_state(self):
        i = np.argmax(self.state, 0) + 1
        j = np.sum(self.state, 0) > 0
        return i * j - 1

    def check_rear(self, color):
        result = np.zeros(self.nl)
        for l in range(self.nl):
            if self.state[color, l, self.cursors[l] - 1] == 1:
                result[l] = self.ll - self.cursors[l]
        return result  # if no matching order, return zeros(6)

    def check_color_block(self, color):
        result = np.ones(self.nl)
        for l in range(self.nl):
            if not self.state[color, l, max(self.cursors[l] - 2,0)]:
                for ll in (self.state[color, l, self.cursors[l] - 2::-1]):
                    row=self.state[color, l, max(self.cursors[l] - 1,0)::-1]
                    row2=self.state[color, l, :]
                    cl=self.cursors[l]
                    if ll:
                        result[l] = 0
                        break
        return result  # if no matching order, return zeros(6)

    def rear_view(self):
        result = np.zeros(self.nl).astype(np.int) - 1
        for l in range(self.nl):
            if self.cursors[l] > 0:
                result[l] = np.argmax(self.state[:, l, self.cursors[l] - 1]).astype(np.int)
        return result

    def front_view(self):
        result = np.zeros(self.nl).astype(np.int) - 1
        for l in range(self.nl):
            if self.cursors[l] > 0:
                result[l] = np.argmax(self.state[:, l, 0]).astype(np.int)
        return result

    def front_hist(self, scope=1):
        result = np.zeros(self.num_of_colors).astype(np.int)
        for l in range(self.nl):
            for i in range (scope):
                if self.cursors[l] > 0:
                    result[np.argmax(self.state[:, l, i]).astype(np.int)] += 1
        return result

    def check_all(self, color):
        result = self.ll - self.cursors.copy()
        for l in range(self.nl):
            for i in range(self.cursors[l] - 1):  # check from 0 to the one before the last one
                if self.state[color, l, i] == 1:
                    result[l] = 0
        return result  # if all lanes contain target order, return zeros(6)

    def insert(self, lane, color=0):

        if self.cursors[lane] < self.ll:
            color = self.in_queue.pop()
            self.state[color, lane, self.cursors[lane]] = 1
            self.cursors[lane] += 1
            return True
        else:
            return False

    def check_cc(self):
        if len(self.out_queue) > 1:
            return not self.out_queue[-2] == self.out_queue[-1]
        else:
            return False

    def release(self, lane):
        if self.cursors[lane] > 0:
            color = np.argmax(self.state[:, lane, 0])
            self.out_queue.append(color)
            self.state[:, lane, :self.cursors[lane]] = self.state[:, lane, 1:self.cursors[lane] + 1]
            self.cursors[lane] -= 1
            return True
        else:
            return False

# for testing

# b=Bank()
# for i in range(5):
#     b.insert(np.random.randint(0,5))
#     print("state:\n{0}\ncursors:\n{1}".format(b.get_view_state()+1, b.cursors))
#     print("In_lane:\n{0}\nOut_lane:\n{1}".format(b.in_queue, b.out_queue))
#
# for i in range(5):
#     b.release(np.random.randint(0, 5))
#     print("state:\n{0}\ncursors:\n{1}".format(b.get_view_state()+1, b.cursors))
#     print("In_lane:\n{0}\nOut_lane:\n{1}".format(b.in_queue, b.out_queue))
