import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    """
    A single sinusoidal layer used by SIREN.
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:

                bound = 1 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:

                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                
    def forward(self, input):

        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    """
    Complete SIREN network with configurable input/output dimensions and hidden-layer structure.
    """
    def __init__(self, in_features, out_features, hidden_features=128, 
                 hidden_layers=3, final_activation=None, omega_0=30, norm_scale=None):
        """
        Parameters:
            in_features (int): Input dimension (for example, 6 for an interplanetary-transfer task)
            out_features (int): Output dimension (for example, 3 for an interplanetary-transfer task)
            hidden_features (int): Number of neurons per hidden layer; defaults to 128
            hidden_layers (int): Number of hidden layers; defaults to 3
            final_activation (nn.Module): Output activation such as nn.Sigmoid(); defaults to None (linear output)
            omega_0 (float): Frequency scaling factor for the sinusoidal activation; defaults to 30
            norm_scale (float, list, tuple or torch.Tensor): Input normalization scale. If provided, the input is divided by this value.
        """
        super().__init__()
        
        if norm_scale is not None:
            if not isinstance(norm_scale, torch.Tensor):
                norm_scale = torch.tensor(norm_scale, dtype=torch.float32)
            self.register_buffer('norm_scale', norm_scale)
        else:
            self.norm_scale = None

        self.net = []
        

        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=omega_0))
        

        for i in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=omega_0))
            

        final_linear = nn.Linear(hidden_features, out_features)
        

        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / omega_0
            final_linear.weight.uniform_(-bound, bound)
            
        self.net.append(final_linear)
        

        if final_activation is not None:
            self.net.append(final_activation)
            
        self.net = nn.Sequential(*self.net)
        
    def forward(self, x):
        if self.norm_scale is not None:
            x = x / self.norm_scale
        return self.net(x)


if __name__ == "__main__":


    model_asteroid = SIREN(in_features=7, out_features=4, hidden_features=128, hidden_layers=3)
    

    model_drone = SIREN(in_features=16, out_features=4, hidden_features=128, 
                        hidden_layers=3, final_activation=nn.Sigmoid())
    

    dummy_input = torch.randn(32, 16) # Batch size 32, 16 features
    output = model_drone(dummy_input)
    print("Output shape:", output.shape)
