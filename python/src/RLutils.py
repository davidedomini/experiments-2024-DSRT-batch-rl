import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def load_neural_network(seed):
    torch.manual_seed(seed)
    model = MLP(20, 128, 10).state_dict()
    return model

def improve_policy(actual_state_batch, action_batch, reward_batch, next_state_batch, seed):
    action_model = ... # TODO - init
    target_model = ... # TODO - init
    gamma = 0.1 # TODO - check

    target_new_state_q_values = target_model(next_state_batch)
    predicted_state_q_values = action_model(actual_state_batch)
    expected_values = (predicted_state_q_values * gamma) + reward_batch

    # TODO - finish