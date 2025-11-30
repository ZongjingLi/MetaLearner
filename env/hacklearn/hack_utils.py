# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 15:14:00
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 15:40:19
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import minihack
import nle
import torch.nn as nn
from nle import nethack
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



def init_render(name = 'Naxx'):
    fig = plt.figure(name, figsize=(10, 3))
    fig.canvas.manager.window.wm_geometry("+300+200")
    return fig

def render_pixels(pixels, delta = 0.01):
    plt.cla()
    plt.imshow(pixels)
    plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.pause(delta)


def get_action_compass():
	return nethack.CompassDirection

from wrapper import NLEWrapper

wrapper = NLEWrapper()



def run_policy(env, policy, render : str = False, delta = 0.1, extra = True):
    """ return the episode reward and action probs etc
    """
    done = False
    episode_reward = 0.0
    obs, info = env.reset()
    init_render()

    actions = [] # the history of action and prob of sample
    rewards = [] # the reward history
    while not done:
        action, logprob = policy.sample(obs)
        prev_obs = obs
        if extra:obs, reward, terminated, truncated, info = env.step(action = action)
        else:
            obs, reward, terminated, truncated = env.step(action = action)
       
        result = wrapper.process_observation(obs)
        #("text_glyphs:", result["text_glyphs"])
        #print(result["text_blstats"])
        #print(result["text_message"])

        episode_reward += reward
        if not done and render:
            pixels = obs["pixel"]
            render_pixels(pixels, delta = delta)
            env.render()
        
        actions.append((action, logprob, prev_obs, obs))
        rewards.append(reward)


        done = terminated or truncated
    #print("epside reward:",episode_reward)
    return {
        "episode_reward": episode_reward,
        "actions" : actions,
        "rewards" : rewards}

def get_normalized_at_location(glyphs):
    # Get the dimensions of the glyphs array
    rows, cols = glyphs.shape
    
    # Iterate through each element to find the '@' character
    for i in range(rows):
        for j in range(cols):
            if chr(glyphs[i, j]) == '@':
                # Normalize the coordinates to [0, 1] range
                norm_row = i / (rows - 1) if rows > 1 else 0.0
                norm_col = j / (cols - 1) if cols > 1 else 0.0
                return (norm_row, norm_col)
    
    # If '@' is not found, return None or handle as needed
    return None

class DummyPolicy(nn.Module):
    """take the obervation and """
    def __init__(self, env):
        super().__init__()
        obs_dim = 2#env.observation_space.shape[0]  # 4
        action_dim = env.action_space.n  # 2
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)  # Output logits
        )

    def forward(self, x):
        return self.net(x)  # Returns logits

    def sample(self, obs):
        msg = ""
        for item in obs["message"]:
            msg += chr(item)
        #print(msg)
        obs = get_normalized_at_location(obs["chars"])
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        logits = self.forward(x)


        dist = Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = torch.log(dist.probs[0,action])  # Log probability (tensor)
        return action, log_prob



class ConvolutionPolicy(nn.Module):
    def __init__(self, env, resolution=(336, 1264, 3)):
        super().__init__()
        self.env = env
        action_dim = self.env.action_space.n
        self.height, self.width, self.channels = resolution
        
        # Convolutional layers: extract spatial features from the image
        # Input shape: (batch_size, 3, 336, 1264) (after channel-first conversion)
        self.conv_layers = nn.Sequential(
            # First conv block: reduce spatial dimensions, increase channels
            nn.Conv2d(in_channels=self.channels, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Stabilize training
            
            # Second conv block: further reduce size
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv block: refine features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth conv block: final downsampling
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate output shape of convolutional layers to size the fully connected layers
        # We'll compute the spatial dimensions after convolutions
        with torch.no_grad():
            # Dummy input to infer output shape
            dummy_input = torch.zeros(1, self.channels, self.height, self.width)
            conv_output = self.conv_layers(dummy_input)
            self.conv_output_size = conv_output.reshape(1, -1).size(1)  # Flattened size
        
        # Fully connected layers: map convolutional features to action logits
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduce overfitting
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)  # Output logits for each action
        )

    def forward(self, x):
        # Input shape: (batch_size, 336, 1264, 3) [H, W, C] (typical for images)
        # Convert to (batch_size, C, H, W) for PyTorch's channel-first convention
        x = x.permute(0, 3, 1, 2)  # Swap dimensions: (B, H, W, C) → (B, C, H, W)
        
        # Extract features with convolutional layers
        x = self.conv_layers(x)
        
        # Flatten convolutional features
        x = x.reshape(x.size(0), -1)  # (B, C, H, W) → (B, C*H*W)
        
        # Map to action logits
        logits = self.fc_layers(x)
        
        return logits
    
    def sample(self, obs):
        """
        Sample an action from the policy distribution and return (action, probability).
        
        Args:
            obs: Observation dictionary containing "pixel" (shape: (336, 1264, 3)).
            
        Returns:
            action: Sampled action (integer, compatible with env.action_space).
            prob: Probability of the sampled action (float).
        """
        # Extract pixel observation and convert to tensor
        pixel_obs = obs["pixel"]  # Shape: (336, 1264, 3)
        x = torch.tensor(pixel_obs, dtype=torch.float32).unsqueeze(0) / 255.  # Convert to tensor
  
        # Get action logits from the network
        logits = self.forward(x)  # Shape: (1, action_dim)
        logits = torch.clamp(logits, min=-10, max=10) 

        # Create a categorical distribution from logits
        #print(logits.sigmoid())
        dist = Categorical(logits = logits)

        # Sample action and get its probability
        action = dist.sample().item()  # Convert to scalar integer
        prob = dist.probs[0, action] # Probability of the sampled action

        return action, torch.log(prob)


class SimplePolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]  # 4
        action_dim = env.action_space.n  # 2
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)  # Output logits
        )

    def forward(self, x):
        return self.net(x)  # Returns logits

    def sample(self, obs):
        # Convert observation to tensor (CartPole obs is a 1D array)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        logits = self.forward(x)

        dist = Categorical(logits=logits)
        action = dist.sample().item()
        log_prob = torch.log(dist.probs[0,action])  # Log probability (tensor)
        return action, log_prob