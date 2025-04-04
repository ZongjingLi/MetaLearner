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
        
        self.max_steps = 5
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
        dense_reward = 1 - np.tanh(distance * 1.0)
        #print(dense_reward)
        
        # Additional rewards for grasping and lifting
        #if self._is_object_grasped():
        #    dense_reward += 1.0
        #    if self._is_object_lifted():
        #        dense_reward += 2.0
        #        print("WOW DONE")
        
        return dense_reward
    
    def _is_object_grasped(self):
        gripper_pos = p.getLinkState(self.sim.robot, self.sim.PANDA_GRIPPER_INDEX)[0]
        target_pos, _ = p.getBasePositionAndOrientation(self.target_object)
        
        distance = np.linalg.norm(np.array(gripper_pos) - np.array(target_pos))
        return distance < 0.05
    
    def _is_object_lifted(self):
        target_pos, _ = p.getBasePositionAndOrientation(self.target_object)
        return target_pos[2] > 0.15
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mean = nn.Linear(64, output_dim)
        self.fc_std = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 0.001  # Ensure positive std
        return mean, std
    
    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.rsample()  # Use reparameterization trick
        action = torch.tanh(action)  # Bound actions to [-1, 1]
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class REINFORCETrainer:
    def __init__(self, env, lr=1e-3, gamma=0.99):
        self.env = env
        self.gamma = gamma
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.policy = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def train(self, num_episodes, batch_size=None):  # batch_size included for interface compatibility
        episode_rewards = []
        from tqdm import tqdm
        for episode in tqdm(range(num_episodes)):
            # Collect trajectory
            states, actions, rewards = [], [], []
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob = self.policy.sample(state_tensor)
                action = action.squeeze(0).detach().numpy()
                
                next_state, reward, done, _ = self.env.step(action)
                
                states.append(state_tensor)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Compute returns and update policy
            returns = self.compute_returns(rewards)
            
            loss = 0
            for state_t, action_t, return_t in zip(states, actions, returns):
                action_t = torch.FloatTensor(action_t).unsqueeze(0)
                _, log_prob = self.policy.sample(state_t)
                loss -= log_prob * return_t
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward (last 100): {avg_reward:.2f}")
        
        return episode_rewards

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_sim = PyBulletSimulator(gui=False)
    train_sim.add_ground()
    train_sim.add_robot_arm()
    train_env = BlockPickingEnv(train_sim)
    trainer = REINFORCETrainer(train_env)
    rewards = trainer.train(num_episodes=int(3000))
    
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