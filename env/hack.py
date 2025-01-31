import torch
import torch.nn as nn
import torch.optim as optim
import gym
import minihack
import numpy as np
from rinarak.logger import get_logger, set_logger_output_file

set_logger_output_file("outputs/hack_outputs.txt")
logger = get_logger("Hack")

# Define the Convolutional Neural Network (CNN) for the state
class CNNPolicyNetwork(nn.Module):
    def __init__(self, input_channels=1, action_space=6):  # Default for MiniHack environments
        super(CNNPolicyNetwork, self).__init__()
        
        # First Convolutional Layer for 'glyphs', 'colors', 'chars'
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(3 * 128 * 21 * 79, 512)  # Flattened from 128x21x79
        self.fc2 = nn.Linear(512, action_space)
        
        self.flatten = nn.Flatten()

    def forward(self, glyphs, colors, chars):
        # Process each input separately
        glyphs = glyphs.to(device)
        colors = colors.to(device)
        chars = chars.to(device)

        glyphs = torch.relu(self.conv1(glyphs))
        glyphs = torch.relu(self.conv2(glyphs))
        glyphs = torch.relu(self.conv3(glyphs))
        
        colors = torch.relu(self.conv1(colors))
        colors = torch.relu(self.conv2(colors))
        colors = torch.relu(self.conv3(colors))
        
        chars = torch.relu(self.conv1(chars))
        chars = torch.relu(self.conv2(chars))
        chars = torch.relu(self.conv3(chars))
        
        # Concatenate all features together
        combined = torch.cat((glyphs, colors, chars), dim=0)  # Concatenate along the channel dimension
        
        # Flatten and pass through fully connected layers
        x = self.flatten(combined[None,...])
        #print(combined.shape)
        #print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x  # Action values (or probabilities, depending on the output activation)

# Initialize the environment
env = gym.make("MiniHack-River-v0")  # You can switch to other environments like "MiniHack-Sokoban2a-v0"

# Create an instance of the policy network
input_channels = 1  # As we have three inputs: glyphs, colors, and chars
action_space = env.action_space.n  # The number of possible actions in the environment
device = "cuda" if torch.cuda.is_available() else "mps"
policy = CNNPolicyNetwork(input_channels=input_channels, action_space=action_space).to(device)
policy.load_state_dict(torch.load("policy_states.pth"))

# Optimizer for training the policy
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Hyperparameters for training
discount_factor = 0.99
epsilon = 0.1  # Epsilon-greedy policy for exploration

num_episodes = 1000
max_timesteps = 32

# Training loop
episode_rewards = []

from tqdm import tqdm

policy = policy.to(device)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    # Convert state components into tensors for the network
    glyphs = torch.tensor(state["glyphs"], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    colors = torch.tensor(state["colors"], dtype=torch.float32).unsqueeze(0)
    chars = torch.tensor(state["chars"], dtype=torch.float32).unsqueeze(0)
    
    for t in range(max_timesteps):
        # Get action from the policy network
        action_values = policy(glyphs, colors, chars)
        action = torch.argmax(action_values).item()  # Use argmax to pick the best action
        
        # Take action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        env.render()
        
        # Convert the next state into tensors
        glyphs_next = torch.tensor(next_state["glyphs"], dtype=torch.float32).unsqueeze(0).to(device)
        colors_next = torch.tensor(next_state["colors"], dtype=torch.float32).unsqueeze(0).to(device)
        chars_next = torch.tensor(next_state["chars"], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Compute the target (reward + discounted future value)
        target = reward + discount_factor * torch.max(policy(glyphs_next, colors_next, chars_next)).item()
        
        # Compute the loss (mean squared error between current Q-value and target)
        loss = (action_values[0, action] - target) ** 2
        
        # Backpropagate and update the policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward += reward
        glyphs, colors, chars = glyphs_next, colors_next, chars_next
        
        if done:
            break
    torch.save(policy.state_dict(), "policy_states.pth")
    
    episode_rewards.append(total_reward)
    
    if episode % 1 == 0:
        logger.info(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")

# Plot reward per episode
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.show()


"""
print(state.keys())
print(state["glyphs"].shape)
print(state["colors"].shape)
print(state["chars"].shape)
print(state["glyphs_crop"].shape)
"""