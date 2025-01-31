import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import robosuite as suite

# Define the policy network (simple feedforward network)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Use tanh to restrict actions within [-1, 1]
        return x

# Create environment instance
env = suite.make(
    env_name="Lift",  # Can be changed to other tasks like "Stack", "Door"
    robots="Panda",  # Can be changed to other robots like "Sawyer", "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Initialize policy network and optimizer
#print(env.observation_spec())
#*env.action_spec[0].shape
input_dim = len(env.observation_spec())  # Observation space dimension
output_dim = len(env.action_spec[0])  # Action space dimension

policy = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# Hyperparameters
gamma = 0.99  # Discount factor
num_episodes = 1000
batch_size = 64

# List to track rewards for each episode
episode_rewards = []

# Function to choose an action based on the policy
def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy(state)
    return action_probs.detach().numpy()

# Main training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    episode_log_probs = []
    episode_rewards_list = []
    
    while not done:
        # Select action from the current policy
        action = select_action(state)
        
        # Take action and observe the next state and reward
        next_state, reward, done, info = env.step(action)
        
        # Track reward and log probabilities (for policy gradient)
        episode_rewards_list.append(reward)
        
        # Store log probability of the action
        log_prob = torch.log(policy(torch.tensor(state, dtype=torch.float32)))
        episode_log_probs.append(log_prob)
        
        # Update state for next iteration
        state = next_state
        episode_reward += reward
    
    # Store the total reward for this episode
    episode_rewards.append(episode_reward)
    
    # Compute the return (discounted sum of future rewards)
    returns = []
    G = 0
    for r in reversed(episode_rewards_list):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Normalize the returns
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)
    
    # Compute loss and update policy
    loss = -torch.stack(episode_log_probs).mul(returns).sum()  # REINFORCE loss
    
    # Perform policy gradient update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print episode information
    if episode % 100 == 0:
        print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward}")

# Optionally, visualize the rewards across episodes
import matplotlib.pyplot as plt
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.show()


#