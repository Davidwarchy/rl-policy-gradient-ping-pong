import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pickle

# Hyperparameters
H = 200  # Number of hidden layer neurons
batch_size = 10  # Every how many episodes to do a parameter update?
learning_rate = 1e-4
gamma = 0.99  # Discount factor for reward
decay_rate = 0.99  # Decay factor for RMSProp leaky sum of grad^2
resume = False  # Resume from previous checkpoint?
render = False  # Render the environment?

# Model definition using PyTorch
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(80*80, H)
        self.affine2 = nn.Linear(H, 1)
    
    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.sigmoid(self.affine2(x))
        return x

# Initialize the policy network
policy = PolicyNet()
if resume:
    policy.load_state_dict(torch.load('policy.pth'))
optimizer = optim.RMSprop(policy.parameters(), lr=learning_rate, weight_decay=decay_rate)

def prepro(I):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # Crop the image
    I = I[::2, ::2, 0]  # Downsample by a factor of 2
    I[I == 144] = 0  # Erase background (background type 1)
    I[I == 109] = 0  # Erase background (background type 2)
    I[I != 0] = 1  # Set paddles and ball to 1
    return I.astype(np.float32).ravel()  # Flatten and convert to float

def discount_rewards(r):
    """ Compute discounted rewards """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # Pong-specific game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    x = torch.from_numpy(x).float()
    aprob = policy(x)
    return aprob

env = gym.make("Pong-v0")
observation = env.reset()
observation = observation[0]
prev_x = None  # Used to compute the difference image
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    # Preprocess the observation, set input to network to be difference image
    I = observation
    cur_x = prepro(I)
    x = cur_x - prev_x if prev_x is not None else np.zeros(80*80)
    prev_x = cur_x

    # Forward the policy network and sample an action from the returned probability
    aprob = policy_forward(x)
    action = 2 if np.random.uniform() < aprob.item() else 3  # Sample action

    # Record intermediates for backprop
    xs.append(x)  # Observations
    y = 1 if action == 2 else 0  # Fake label
    dlogps.append(y - aprob.item())  # Gradients that encourage the action taken

    # Step the environment
    observation, reward, done, truncated, info = env.step(action)
    reward_sum += reward
    drs.append(reward)  # Record reward

    if done:  # Episode finished
        episode_number += 1

        # Stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = torch.stack([torch.from_numpy(x) for x in xs]).float()
        epdlogp = torch.tensor(dlogps, dtype=torch.float32)
        epr = np.vstack(drs)
        xs, dlogps, drs = [], [], []  # Reset array memory

        # Compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        discounted_epr = torch.tensor(discounted_epr, dtype=torch.float32).squeeze()

        # Compute loss
        aprobs = policy(epx).squeeze()
        loss = -(torch.log(aprobs) * epdlogp * discounted_epr).sum()

        # Backpropagate the gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))

        if episode_number % 100 == 0:
            torch.save(policy.state_dict(), 'policy.pth')

        reward_sum = 0
        observation = env.reset()  # Reset environment
        observation = observation[0]
        prev_x = None

    if reward != 0:  # Pong has +1 or -1 reward at the end of a game
        print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
