
import torch
from tqdm import tqdm

class Reinforce:
    def __init__(self,  model, env, run_policy, gamma = 0.99, extra = True):
        self.run_policy = run_policy
        self.model = model
        self.env   = env
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3)
        self.extra = extra

    def discount_returns(self, rewards):
        """return a sequence of rewards"""
        discount_rewards = []
        reward = 0
        for rwd in reversed(rewards):
            reward = rwd + self.gamma * reward
            discount_rewards.insert(0,reward)
        discount_rewards = torch.tensor(discount_rewards).float()
        std  = torch.std(discount_rewards) + 1e-5
        mean = torch.mean(discount_rewards)
        discount_rewards = (discount_rewards - mean)/std

        return discount_rewards

    def update_policy(self, result):
        rewards = result["rewards"]
        discount_rewards = self.discount_returns(rewards)
        #action_probs = [bind[1] for bind in result["actions"]]

        action_log_probs = [bind[1] for bind in result["actions"]]  # List of tensors
        log_probs_tensor = torch.stack(action_log_probs)
        loss = -torch.mean(log_probs_tensor * discount_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, episodes = 1000, delta = 0.1):
        episode_rewards = []
        losses = []
        for episode in tqdm(range(episodes)):
            self.env.reset()
            results = self.run_policy(self.env, self.model, delta = delta, render = False, extra = self.extra)
            episode_rewards.append(results["episode_reward"])
            loss = self.update_policy(results)
            losses.append(loss)

        return episode_rewards