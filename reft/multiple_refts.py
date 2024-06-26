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
import pyreft
from pyreft import get_reft_model , ReftConfig , LoreftIntervention , ReftTrainerForCausalLM
import transformers

device = "cuda"

model_name_or_path = "/model-weights/Meta-Llama-3-8B-Instruct"

# Load the base model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

# Get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

# First ReFT model
reft_config = pyreft.ReftConfig(representations={
    "layer": 15,
    "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
                                               low_rank_dimension=4)
})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
print("First ReFT Model:")
reft_model.print_trainable_parameters()

# Second ReFT model based on the first ReFT model
second_reft_config = pyreft.ReftConfig(representations={
    "layer": 10,
    "component": "block_output",
    "low_rank_dimension": 2,
    "intervention": pyreft.LoreftIntervention(embed_dim=reft_model.model.config.hidden_size,
                                               low_rank_dimension=2)
})
second_reft_model = pyreft.get_reft_model(reft_model.model, second_reft_config)
second_reft_model.set_device("cuda")
print("\nSecond ReFT Model:")
second_reft_model.print_trainable_parameters()

# Third ReFT model based on the second ReFT model
third_reft_config = pyreft.ReftConfig(representations={
    "layer": 12,
    "component": "block_output",
    "low_rank_dimension": 2,
    "intervention": pyreft.LoreftIntervention(embed_dim=second_reft_model.model.config.hidden_size,
                                               low_rank_dimension=2)
})
third_reft_model = pyreft.get_reft_model(second_reft_model.model, third_reft_config)
third_reft_model.set_device("cuda")
print("\nThird ReFT Model:")
third_reft_model.print_trainable_parameters()


