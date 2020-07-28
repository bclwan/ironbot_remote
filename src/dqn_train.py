#!/usr/bin/env python3
import wrapper
import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt



DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 150

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 600

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 0.70
EPSILON_FINAL = 0.02

STATE_W = 84
STATE_H = 84

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states).reshape(batch_size, 1, STATE_W, STATE_H), \
               np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states).reshape(batch_size, 1, STATE_W, STATE_H)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            try:
                print("Exploitation")
                if self.state.shape[0]!=84 or self.state.shape[1]!=84:
                    print("State shape violation! ", self.state.shape)
                    action = 0
                else:
                    state_a = np.array([self.state], copy=False)
                    state_v = torch.tensor(state_a).to(device)
                    state_v = state_v.unsqueeze(0)
                    q_vals_v = net(state_v.float())
                    _, act_v = torch.max(q_vals_v, dim=1)
                    action = int(act_v.item())
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print(state_v.shape)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        #print(type(exp[0]))
        #print(exp[0].shape)

        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    """
    plt.imshow(states[0].reshape(STATE_W, STATE_H))
    plt.show()
    plt.imshow(next_states[0].reshape(STATE_W, STATE_H))
    plt.show()
    """

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    #states_v = states_v.unsqueeze(0)
    #next_states_v = next_states_v.unsqueeze(0)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrapper.make_env(args.env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    tgt_net.load_state_dict(torch.load("/home/bwan/Projects/ROSWS/src/ironbot_remote/src/dqn_self_driving.dat"))
    net.load_state_dict(torch.load("/home/bwan/Projects/ROSWS/src/ironbot_remote/src/dqn_self_driving.dat"))

    #Try Network in/out
    """
    print("Test Network In/Out")
    sample = np.ones(env.observation_space.shape, dtype=np.float)/255
    sample_ts = torch.tensor(sample).to(device)
    sample_ts = sample_ts.unsqueeze(0)
    print("In: ", sample_ts.shape)
    example_out = net(sample_ts.float())
    print("Out: ", example_out.shape)
    """

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    test_mode = False

    if test_mode:
        while True:
            reward = agent.play_step(net, 0, device=device)
            if reward is not None:
                print("Played one game")
                print("Total Reward: ", reward)
                print("Collision Flag: ", env.coll_cnt_sum)
                break
            
    else:
        while True:
            frame_idx += 1
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                mean_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), mean_reward, epsilon,
                    speed
                ))
                
                torch.save(net.state_dict(), "/home/bwan/Projects/ROSWS/src/ironbot_remote/src/model/training_backup.dat")
                print("Saved training backup")

                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    torch.save(net.state_dict(), args.env + "-best.dat")
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                if mean_reward > args.reward:
                    print("Solved in %d frames!" % frame_idx)
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                print("Target Network update")
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            print(loss_t)

            optimizer.step()
        writer.close()
