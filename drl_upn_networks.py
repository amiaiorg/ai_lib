
import torch
import torch.nn as nn

class DRLNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DRLNetwork, self).__init__()
        self.resonance_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.pattern_memory = {}
        
    def forward(self, x, resonance_pattern=None):
        if resonance_pattern is not None:
            x = x * resonance_pattern
        return self.resonance_layer(x)
    
    def store_pattern(self, pattern_id, pattern):
        self.pattern_memory[pattern_id] = pattern

class UPNNetwork(nn.Module):
    def __init__(self, input_size, pattern_size):
        super(UPNNetwork, self).__init__()
        self.pattern_generator = nn.Sequential(
            nn.Linear(input_size, pattern_size * 2),
            nn.ReLU(),
            nn.Linear(pattern_size * 2, pattern_size),
            nn.Tanh()
        )
        
    def forward(self, user_input):
        return self.pattern_generator(user_input)
    
    def generate_pattern(self, user_data):
        with torch.no_grad():
            return self.forward(user_data)
