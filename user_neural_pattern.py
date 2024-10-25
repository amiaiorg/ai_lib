
import torch
import torch.nn as nn

class UserNeuralPattern(nn.Module):
    def __init__(self, input_dim, pattern_dim):
        super(UserNeuralPattern, self).__init__()
        self.pattern_layer = nn.Sequential(
            nn.Linear(input_dim, pattern_dim),
            nn.ReLU(),
            nn.Linear(pattern_dim, pattern_dim // 2),
            nn.Tanh()
        )
        
    def forward(self, user_input):
        return self.pattern_layer(user_input)

# Example usage
input_dim = 128
pattern_dim = 64
user_pattern = UserNeuralPattern(input_dim, pattern_dim)

# Generate a pattern for a given user input
user_input = torch.randn(input_dim)
pattern = user_pattern(user_input)
