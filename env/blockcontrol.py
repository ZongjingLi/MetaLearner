import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pybullet as p
from collections import deque
import random
import time

from blockgripper_env import *


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

class BlockPickingEnv(gym.Env):
    def __init__(self, simulator):
        super().__init__()
        self.sim = simulator
        
        # Define action space (dx, dy, dz, dgripper)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        self.max_steps = 100
        self.current_step = 0
        self.target_object = None
        self.unique = 1
    
    def reset(self):
        self.current_step = 0
        self.sim.reset_arm()
        
        if self.unique:
            x = np.random.uniform(0.6, 0.6)
            y = np.random.uniform(-0.3, 0.3)
            self.target_object = self.sim.add_box([x, y, 0.1])
            self.unique = 0
        
        return self._get_observation()
    
    def _get_observation(self):
        ee_state = p.getLinkState(self.sim.robot, self.sim.PANDA_EE_INDEX)
        ee_pos = ee_state[0]
        
        gripper_state = p.getJointState(self.sim.robot, self.sim.PANDA_GRIPPER_INDEX)[0]
        
        target_pos, _ = p.getBasePositionAndOrientation(self.target_object)
        
        relative_pos = np.array(target_pos) - np.array(ee_pos)
        
        obs = np.concatenate([
            ee_pos,
            [gripper_state],
            target_pos,
            relative_pos
        ]).astype(np.float32)
        
        return obs
    
    def step(self, action):
        self.current_step += 1
        
        # Scale actions from [-1, 1] to actual ranges
        dx = action[0] * 0.1  # Scale to ±0.1m
        dy = action[1] * 0.1
        dz = action[2] * 0.1
        dgripper = action[3]  # Already in [-1, 1]
        
        # Get current end effector position
        ee_pos = p.getLinkState(self.sim.robot, self.sim.PANDA_EE_INDEX)[0]
        
        # Calculate new target position
        new_pos = [
            ee_pos[0] + dx,
            ee_pos[1] + dy,
            ee_pos[2] + dz
        ]
        
        # Move arm
        self.sim.move_arm(
            new_pos,
            p.getQuaternionFromEuler([0, np.pi, 0])
        )
        
        # Control gripper
        self.sim.control_gripper(open=bool(dgripper > 0))
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self.compute_dense_reward(obs)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return obs, reward, done, {}
    
    def compute_dense_reward(self, obs):
        ee_pos = obs[:3]
        target_pos = obs[4:7]
        
        # Calculate distance
        distance = np.linalg.norm(ee_pos - target_pos)
        
        # Dense reward similar to the paper: 1 - tanh(10.0 * d)
        dense_reward = 1 - np.tanh(distance * 10.0)
        #print(dense_reward)
        
        # Additional rewards for grasping and lifting
        if self._is_object_grasped():
            dense_reward += 1.0
            if self._is_object_lifted():
                dense_reward += 2.0
                print("WOW DONE")
        
        return dense_reward
    
    def _is_object_grasped(self):
        gripper_pos = p.getLinkState(self.sim.robot, self.sim.PANDA_GRIPPER_INDEX)[0]
        target_pos, _ = p.getBasePositionAndOrientation(self.target_object)
        
        distance = np.linalg.norm(np.array(gripper_pos) - np.array(target_pos))
        return distance < 0.05
    
    def _is_object_lifted(self):
        target_pos, _ = p.getBasePositionAndOrientation(self.target_object)
        return target_pos[2] > 0.15

class SACPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        action = action.squeeze(1)  # Ensure action is 2D like state
        #print(action.shape,state.shape)
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(args)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return [torch.FloatTensor(x) for x in zip(*batch)]
    
    def __len__(self):
        return len(self.buffer)

class SACTrainer:
    def __init__(self, env, hidden_dim=128, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.env = env
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        
        self.policy = SACPolicy(obs_dim, action_dim, hidden_dim)
        self.q_net = QNetwork(obs_dim, action_dim, hidden_dim)
        self.q_net_target = QNetwork(obs_dim, action_dim, hidden_dim)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
    
    def train_step(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state)
            q1_target, q2_target = self.q_net_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            y = reward + (1 - done) * self.gamma * q_target
        
        q1, q2 = self.q_net(state, action)
        q_loss = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        new_action, log_prob = self.policy.sample(state)
        q1_new, q2_new = self.q_net(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, num_episodes=1000, batch_size=256):
        rewards = []
        for episode in range(num_episodes):
            state, episode_reward, done = self.env.reset(), 0, False
            while not done:
                with torch.no_grad():
                    action, _ = self.policy.sample(torch.FloatTensor(state).unsqueeze(0))
                state, reward, done, _ = self.env.step(action.squeeze(0).numpy())
                self.replay_buffer.push(state, action.numpy(), reward, state, done)
                self.train_step(batch_size)
                episode_reward += reward
            rewards.append(episode_reward)
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")
        return rewards

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_sim = PyBulletSimulator(gui=False)
    train_sim.add_ground()
    train_sim.add_robot_arm()
    train_env = BlockPickingEnv(train_sim)
    trainer = SACTrainer(train_env)
    rewards = trainer.train(num_episodes=int(1e5), batch_size=256)
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()
    train_sim.close()
    
    print("Testing policy...")
    test_sim = PyBulletSimulator(gui=True)
    test_sim.add_ground()
    test_sim.add_robot_arm()
    test_env = BlockPickingEnv(test_sim)
    
    state, total_reward, done = test_env.reset(), 0, False
    while not done:
        with torch.no_grad():
            action, _ = trainer.policy.sample(torch.FloatTensor(state).unsqueeze(0))
        state, reward, done, _ = test_env.step(action.squeeze(0).numpy())
        total_reward += reward
        time.sleep(0.01)
    print(f"Test completed with total reward: {total_reward:.2f}")
    test_sim.close()

