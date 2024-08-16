# README: Pong Policy Gradient Agent

## Overview
This repository contains a simple implementation of a Policy Gradient agent designed to play Pong using the OpenAI Gym environment. The code is based on Andrej Karpathy's blog post ["Deep Reinforcement Learning: Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/), with minor adjustments for Python 3 compatibility.

The agent uses a simple neural network with one hidden layer to approximate the policy, and it is trained using the REINFORCE algorithm (also known as Monte Carlo Policy Gradient). The neural network updates its weights to maximize the probability of taking actions that lead to higher rewards, effectively learning to improve its Pong gameplay over time.

## Files
- `pong_policy_gradient.py`: The main Python script implementing the agent.
- `save.p`: A checkpoint file used to save the model's parameters periodically.

## Requirements
- Python 3.x
- OpenAI Gym
- NumPy

You can install the required packages via pip:
```bash
pip install gym numpy
```

## How It Works
1. **Preprocessing**: Each frame of the Pong game is preprocessed to reduce the dimensionality. The frame is cropped, downsampled, and converted to a binary image, reducing the size to an 80x80 grid of pixels.

2. **Policy Network**: The agent uses a simple feedforward neural network with one hidden layer. The network consists of:
   - Input layer: A flattened 80x80 grid (6,400-dimensional input).
   - Hidden layer: 200 neurons with ReLU activation.
   - Output layer: A single output representing the probability of taking action 2 (up) or action 3 (down).

3. **Policy Gradient**: The agent uses the REINFORCE algorithm to compute gradients and update the network's weights. Rewards are discounted over time, and the gradient is modulated by the advantage function (which is the discounted reward in this case). RMSProp is used to optimize the parameters.

4. **Training**: The agent plays the game, collects rewards, and after every `batch_size` episodes, it updates the network weights based on the accumulated gradients.

5. **Saving and Loading**: The model is periodically saved to a file (`save.p`) to allow for resuming training from a previous checkpoint.

## Hyperparameters
- `H`: Number of neurons in the hidden layer (default: 200)
- `batch_size`: Number of episodes after which parameters are updated (default: 10)
- `learning_rate`: Learning rate for the RMSProp optimizer (default: 1e-4)
- `gamma`: Discount factor for future rewards (default: 0.99)
- `decay_rate`: Decay factor for RMSProp (default: 0.99)
- `resume`: If True, resume training from the last saved checkpoint (default: False)
- `render`: If True, render the Pong environment during training (default: False)

## Usage
To train the agent from scratch or resume from a previous checkpoint, run the script:
```bash
python pong_policy_gradient.py
```

You can enable rendering by setting `render = True` in the script to visually see the agent playing Pong.

## Acknowledgments
This implementation is based on the work of Andrej Karpathy and his blog post ["Deep Reinforcement Learning: Pong from Pixels"](https://karpathy.github.io/2016/05/31/rl/).