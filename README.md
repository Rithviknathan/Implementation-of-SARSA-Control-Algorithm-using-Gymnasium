# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement



## Software Requirements



## Environment Description
Frozenlake
```
    "FSFF",
    "FHFH",
    "FFFG",
    "HFFH"
```
## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm


## Python Program

```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np

custom_map = [
    "FSFF",
    "FHFH",
    "FFFG",
    "HFFH"
]

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False, 
    render_mode="rgb_array", 
    desc=custom_map
)

# Initialize environment for SARSA
obs_space_size = env.observation_space.n  
action_space_size = env.action_space.n    

# Initialize Q-table
Q = np.zeros((obs_space_size, action_space_size))

print("Custom FrozenLake Environment Created!")
print("Map Layout:")
for row in custom_map:
    print(row)

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          
gamma = 0.9         

epsilon = 1.0        
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          
gamma = 0.9         

epsilon = 1.0        
epsilon_min = 0.05
epsilon_decay = 0.9995

# -------------------------------------------------
# Initialize Q-table
# This has already been done in cell 2.
# Q = np.zeros((obs_space_size, action_space_size))
# -------------------------------------------------



# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return env.action_space.sample()
    else:
        # Exploit: choose the action with the highest Q-value for the current state
        return np.argmax(Q[state, :])

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset() # Reset the environment for a new episode
    action = epsilon_greedy_action(state, epsilon) # Select initial action
    total_reward = 0

    for step in range(max_steps_per_episode):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Select the next action using epsilon-greedy policy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)


# Derive the optimal policy and state values from the learned Q-table
learned_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

print("SARSA training complete!")

# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()

```
---

## Output


Final Q-table:
[[0.627 0.395 0.818 0.557]
 [0.808 0.    0.932 0.867]
 [0.815 0.922 0.833 0.843]
 [0.892 0.    0.74  0.724]
 [0.077 0.631 0.    0.131]
 [0.    0.    0.    0.   ]
 [0.    0.99  0.    0.95 ]
 [0.    0.    0.    0.   ]
 [0.126 0.    0.872 0.196]
 [0.318 0.409 0.988 0.   ]
 [0.967 0.976 1.    0.969]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.067 0.464 0.775]
 [0.443 0.774 0.    0.989]
 [0.    0.    0.    0.   ]]





Estimated State-Value Function:
[[0.818 0.932 0.922 0.892]
 [0.631 0.    0.99  0.   ]
 [0.872 0.988 1.    0.   ]
 [0.    0.775 0.989 0.   ]]




Learned Policy:
[['R' 'R' 'D' 'L']
 ['D' 'L' 'D' 'L']
 ['R' 'R' 'R' 'L']
 ['L' 'U' 'U' 'L']]



Average reward over last 1000 episodes: 0.96


```

---

## Result
```text



```

---

## Inference
```text



```
---

