import numpy as np
import pickle
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing as mp
from numba import jit, float64, int64


# Hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
num_processes = 4  # number of parallel processes

# Model definition using PyTorch
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

# Initialize model and optimizer
D = 80 * 80
model = PolicyNetwork(D, H, 1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

@jit(nopython=True)
def prepro(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195:2, ::2, 0]  # downsample by factor of 2
    out = np.zeros_like(I, dtype=np.float32)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i, j] == 144 or I[i, j] == 109:
                out[i, j] = 0
            elif I[i, j] != 0:
                out[i, j] = 1
    return out.ravel()

@jit(nopython=True)
def discount_rewards(r, gamma=0.99):
    n = len(r)
    discounted_r = np.zeros(n, dtype=np.float64)
    running_add = 0.0
    for t in range(n - 1, -1, -1):
        if r[t] != 0:
            running_add = 0.0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def run_episode(env):
    observation, _ = env.reset()
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    done = False
    reward_sum = 0
    
    while not done:
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        
        # Forward pass
        with torch.no_grad():
            aprob = model(torch.from_numpy(x).float().unsqueeze(0))
        action = 2 if np.random.uniform() < aprob.item() else 3
        
        # Record data
        xs.append(x)
        y = 1 if action == 2 else 0
        dlogps.append(y - aprob.item())
        
        # Step environment
        observation, reward, done, truncated, _ = env.step(action)
        reward_sum += reward
        drs.append(reward)
        
        if done or truncated:
            break
    
    return np.vstack(xs), np.array(dlogps), np.array(drs), reward_sum

def train():
    env = gym.make("ALE/Pong-v5")
    running_reward = None
    episode_number = 0
    
    while True:
        pool = mp.Pool(processes=num_processes)
        results = [pool.apply_async(run_episode, (env,)) for _ in range(batch_size)]
        episodes = [p.get() for p in results]
        pool.close()
        pool.join()
        
        episode_number += batch_size
        
        # Combine episodes
        combined_xs = np.concatenate([ep[0] for ep in episodes])
        combined_dlogps = np.concatenate([ep[1] for ep in episodes])
        combined_drs = np.concatenate([ep[2] for ep in episodes])
        reward_sums = [ep[3] for ep in episodes]
        
        # Compute discounted rewards
        discounted_epr = np.concatenate([discount_rewards(ep_rewards) for ep_rewards in combined_drs])
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        
        # Compute loss and update model
        optimizer.zero_grad()
        probs = model(torch.from_numpy(combined_xs).float())
        loss = -torch.sum(torch.from_numpy(combined_dlogps).float() * probs.squeeze() * torch.from_numpy(discounted_epr).float())
        loss.backward()
        optimizer.step()
        
        # Book-keeping
        running_reward = np.mean(reward_sums) if running_reward is None else running_reward * 0.99 + np.mean(reward_sums) * 0.01
        print(f'Batch {episode_number}: mean reward: {np.mean(reward_sums):.3f}, running mean: {running_reward:.3f}')
        
        if episode_number % 100 == 0:
            torch.save(model.state_dict(), 'pong_model.pth')

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Add this line
    train()