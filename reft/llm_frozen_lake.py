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
import transformers
import pyreft

try:
    os.environ["DISPLAY"]
except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


import pdb

def convert_to_dtype(tensor, dtype):
    if tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor

# Define Q-Network using LLM
class LLMQN(nn.Module):
    def __init__(self, llm, num_actions, reft_config=None):
        super().__init__()
        self.llm = llm
        self.num_actions = num_actions
        self.config = reft_config
        
        if reft_config:
            self.reft_model = pyreft.get_reft_model(self.llm, reft_config)
            self.reft_model.set_device("cuda")
        else:
            self.reft_model = None

        self.q_head = nn.Linear(self.llm.config.hidden_size, num_actions).to("cuda")

        self.dtype = next(llm.parameters()).dtype
        self.q_head = self.q_head.to(self.dtype)

    def forward(self, tokenized_state):
        tokenized_state = {k: v.to("cuda") for k, v in tokenized_state.items()}
        # print(f"tokenized_state: {tokenized_state}")
        
        if self.reft_model:
            # outputs = self.reft_model(**tokenized_state, output_hidden_states=True)
            
            outputs = self.reft_model(tokenized_state["input_ids"], output_hidden_states=True)
        else:
            outputs = self.llm(**tokenized_state, output_hidden_states=True)
        
        cls_token = outputs.hidden_states[-1][:, -1, :].to("cuda") # last layer, last token

        cls_token = convert_to_dtype(cls_token, self.dtype)
        
        q_values = self.q_head(cls_token)
        
        return q_values


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
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000       # size of replay memory
    mini_batch_size = 32 # 32            # size of the training data set sampled from the replay memory

    # loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    def __init__(self, model, tokenizer, reft_config=None):
        self.model = model
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.reft_config = reft_config

    def train(self, num_episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = LLMQN(self.model, num_actions, self.reft_config)
        target_dqn = LLMQN(self.model, num_actions, self.reft_config)
        
        count_parameters(policy_dqn)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        rewards_per_episode = np.zeros(num_episodes)

        epsilon_history = []

        # For syncing the target and policy dqn
        step_count=0
        train_iter=0
        
        for i in range(num_episodes):
            # Start a new episode
            
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent collect a trajectory
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # print(f"Current state: {state}")

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # print(f"--Selecting action randomly--")
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                    # print(f"Selected {action}")
                else:
                    # select best action  
                    # print(f"--Selecting action using policy--")          
                    with torch.no_grad():
                        # import pdb; pdb.set_trace()
                        tokenized_state = self.tokenize_state(state)
                        # print(f"state: {state}; tokenized state: {tokenized_state}")
                        action = policy_dqn(tokenized_state).argmax().item()
                        # print(f"Selected {action}")
                
                # print('-' * 30)

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
                train_iter += 1
                # print(f"Train iteration: {train_iter}")

                # Decay epsilon
                epsilon = max(epsilon - 1/num_episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0


        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "./frozen_lake_dql_reft.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(num_episodes)
        for x in range(num_episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('./frozen_lake_dql_reft.png')

            
    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []
        terminated_batch = []
        
        # Create a list for each of state, action, new_state, reward, terminated
        
        for (state, action, new_state, reward, terminated) in mini_batch:
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
            terminated_batch.append(terminated)
        
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.model.device).to(torch.bfloat16)
        terminated_batch = torch.tensor(terminated_batch, dtype=torch.float).to(self.model.device).to(torch.bfloat16)
        new_state_batch = torch.tensor(new_state_batch, dtype=torch.float).to(self.model.device).to(torch.bfloat16)


        # Tokenize state and new state batches
        state_batch = [self.tokenize_state(state) for state in state_batch]
        new_state_batch = [self.tokenize_state(state) for state in new_state_batch]

        # create dictionary: input_ids, attention_mask 
        state_batch = self.stack_tokenized_inputs(state_batch)
        new_state_batch = self.stack_tokenized_inputs(new_state_batch)

        # print(f"Calculating q_values ..")
        q_values = policy_dqn(state_batch).gather(1, torch.tensor(action_batch).unsqueeze(-1).to(self.model.device)).squeeze()
        # print(f"q_values: {q_values}")
        with torch.no_grad():
            # print(f"Calculating target_q_values ..")
            # Bellman equation Q learning
            target_q_values = reward_batch + (1 - terminated_batch) * self.discount_factor_g * target_dqn(new_state_batch).max(1)[0]
            target_q_values = target_q_values.to(torch.bfloat16)
            # print(f"target_q_values: {target_q_values}")
        
        loss = self.loss_fn(q_values, target_q_values)
        print(loss)
        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        # import pdb; pdb.set_trace()
        self.optimizer.step()
    

    def tokenize_state(self, state):
        """ Format input as 'state: {state}' and tokenize it
        """
        # state_str = f"state: {state}"
        state_str = f"The game is frozen lake with a 4 by 4 grid. Each cell is labeled as an integer from 0 to 15. We are currently at the cell: {state}. Give me the next action so that I can reach the goal 15 as fastest as possible. left = 0\ndown = 1\nright = 2\nup = 3.\nOutput the corresponding integer"
        inputs = self.tokenizer(state_str, return_tensors="pt")
        return inputs.to(self.model.device)
    

    def stack_tokenized_inputs(self, tokenized_inputs):
        """ Stack tokenized inputs into dictionary : input_ids, attention_mask
        """
        input_ids = torch.cat([ti["input_ids"] for ti in tokenized_inputs], dim=0)
        attention_mask = torch.cat([ti["attention_mask"] for ti in tokenized_inputs], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


    def print_dqn(self, dqn):
        for s in range(16):
            state = self.tokenize_state(s)
            with torch.no_grad():
                action = dqn(state).argmax().item()
                # print(f'{s}: {self.ACTIONS[action]}')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"Trainable parameters: {trainable_params} || Total parameters: {total_params}")
    # print(f"Trainable %: {(trainable_params / total_params) * 100}")


if __name__ == '__main__':
    model_name_or_path = "/model-weights/Meta-Llama-3-8B-Instruct"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    # import pdb; pdb.set_trace()

    # Configure REFT
    reft_config = pyreft.ReftConfig(representations={
        "layer": 15, 
        "component": "block_output",
        "low_rank_dimension": 4,
        "intervention": pyreft.LoreftIntervention(
            embed_dim=model.config.hidden_size,
            low_rank_dimension=4
        )
    })

    dql = FrozenLakeDQL(model, tokenizer, reft_config=None)
    dql.train(num_episodes=1000)
