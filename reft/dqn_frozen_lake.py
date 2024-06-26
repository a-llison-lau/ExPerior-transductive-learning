# srun --mem=48GB -c 4 --gres=gpu:1 --time=3:00:00 --qos=normal -p a40 -n 1 --pty bash

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import os
import time

try:
    os.environ["DISPLAY"]
except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# DQN
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)   
        self.out = nn.Linear(h1_nodes, out_actions) 

    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = self.out(x)         
        return x


# Define DQN with LoReFT
class DQN_LoReFT(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions, r):
        super().__init__()
        
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)
        
        # LoReFT parameters
        self.r = r
        self.R = nn.Parameter(torch.randn(r, h1_nodes)) # (2, 16)
        self.W = nn.Parameter(torch.randn(r, h1_nodes)) # (2, 16)
        self.b = nn.Parameter(torch.randn(r))
        
        # Ensure R has orthonormal rows
        self._orthonormalize_R()

        # Freeze other layers
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False

    def _orthonormalize_R(self):
        with torch.no_grad():
            # Gram-Schmidt orthogonalization
            for i in range(self.R.shape[0]):
                for j in range(i):
                    self.R.data[i] -= torch.dot(self.R.data[i], self.R.data[j]) * self.R.data[j]
                self.R.data[i] /= torch.norm(self.R.data[i])

    def forward(self, x, node_idx=None):
        node_idx = [1, 13]

        # print(f"x: {x}")
        # print(f"x shape: {x.shape}") # (16,)

        # Forward pass through the first layer with ReLU activation
        h = F.relu(self.fc1(x))
        # print(f"h: {h}")
        # print(f"h shape: {h.shape}") # (16,)

        self._orthonormalize_R()

        # Create a mask of zeros
        mask = torch.zeros_like(h)
        mask[node_idx] = 1
        masked_h = (h * mask).unsqueeze(1) # (16, 1)
        # print(f"masked_h: {masked_h}")
        
        # LoReFT intervention
        R_h = torch.matmul(self.R, masked_h)  # (r, 1)
        Wh_b = torch.matmul(self.W, masked_h) + self.b.unsqueeze(1)  # (r, 1)
        intervention = (masked_h + torch.matmul(self.R.T, (Wh_b - R_h))).squeeze()  # (h1_nodes, 1)
        
        # Update the hidden representation with the intervention
        h[node_idx] = intervention[node_idx]
        
        # Forward pass through the output layer
        x = self.out(h)
        return x
    
    
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('=' * 60)
    print(f"Number of trainable params: {trainable_params} || Total number of params: {total_params}")
    print(f"Percentage: {(trainable_params / total_params) * 100} %")
    print('=' * 60)

     
# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes, render=False, is_slippery=False, reft=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1 # 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        if not reft:
            policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
            target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        else:
            policy_dqn = DQN_LoReFT(in_states=num_states, h1_nodes=num_states, out_actions=num_actions, r=2)
            target_dqn = DQN_LoReFT(in_states=num_states, h1_nodes=num_states, out_actions=num_actions, r=2)
        
        count_parameters(policy_dqn)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
        
        
        start_time = time.time()
        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent collect a trajectory
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode)>0:
                mini_batch = memory.sample(self.mini_batch_size)
                # Compute loss using policy dqn and target dqn
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        end_time = time.time()
        print(f"Training time: {end_time - start_time}")

        # Close environment
        env.close()

        # Save policy
        if reft: 
            torch.save(policy_dqn.state_dict(), "./frozen_lake_dql_reft.pt")
        else: 
            torch.save(policy_dqn.state_dict(), "./frozen_lake_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        if reft:
            plt.savefig('./frozen_lake_dql_reft.png')
        else:
            plt.savefig('./frozen_lake_dql.png')
            

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False, reft=True):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        if reft:
            policy_dqn = DQN_LoReFT(in_states=num_states, h1_nodes=num_states, out_actions=num_actions, r=2) 
            policy_dqn.load_state_dict(torch.load("./frozen_lake_dql_reft.pt"))
        else:
            policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
            policy_dqn.load_state_dict(torch.load("./frozen_lake_dql.pt"))

        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':
    # Seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    reft = False
    frozen_lake.train(1000, is_slippery=is_slippery, reft=reft)
    frozen_lake.test(10, is_slippery=is_slippery, reft=reft)