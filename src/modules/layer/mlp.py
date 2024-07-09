import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers=3, 
                 activation=nn.ReLU(), 
                 output_activation=nn.Identity(), 
                 init_method=None, 
                 bias=True):
        
        super(MLP, self).__init__()
        
        layers = []
        in_size = input_dim
        for _ in range(n_layers - 1):
            curr_layer = nn.Linear(in_size, hidden_dim)
            if init_method is not None:
                curr_layer.apply(init_method)
            layers.append(curr_layer)
            layers.append(activation)
            in_size = hidden_dim

        last_layer = nn.Linear(in_size, output_dim, bias=bias)
        if init_method is not None:
            last_layer.apply(init_method)
        layers.append(last_layer)
        layers.append(output_activation)

        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        q = self.mlp(inputs)
        return q
