"""Base ResNet architecture."""

import torch 
import torch.nn as nn
import torch.nn.functional as fn

class Reducer(nn.Module):
    """Reduce feature dimensions."""

    def __init__(self, in_dim=512, out_dim=64):
        """Initialize weights."""
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        """Forward pass."""
        return self.layer(x)


class ActivationLinear(nn.Module):
    def __init__(self, delta, dim=64, activation=fn.tanh,
                 max_initial_norm=0.1, ntrunc:bool=False):
        """Initialize weights."""
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(dim, dim, bias=False)
        self.delta = delta

        if ntrunc and abs(max_initial_norm) > 1e-9:
            nn.init.trunc_normal_(self.layer.weight, 
                                    mean = 0.0,
                                    std = max_initial_norm ** 0.5, 
                                    a   = - max_initial_norm, 
                                    b   = max_initial_norm) 
        else:
            nn.init.constant_(self.layer.weight, val=max_initial_norm)


    def forward(self, x):
        """Forward pass."""
        return self.delta * self.activation(self.layer(x))


class Block(nn.Module):
    """Single ResNet block with skip connection."""

    def __init__(self, delta, dim=64, activation=fn.tanh,
                 max_initial_norm=0.1, ntrunc:bool=False):
        """Initialize weights."""
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(dim, dim, bias=False)
        self.delta = delta

        if ntrunc and abs(max_initial_norm) > 1e-9:
            nn.init.trunc_normal_(self.layer.weight, 
                                    mean = 0.0,
                                    std = max_initial_norm ** 0.5, 
                                    a   = - max_initial_norm, 
                                    b   = max_initial_norm) 
        else:
            nn.init.constant_(self.layer.weight, val=max_initial_norm)

    def forward(self, x):
        """Forward pass."""
        return x + self.delta * self.activation(self.layer(x))


class IncSkipBlock(nn.Module):
    """Single ResNet block with skip noisy connection."""

    def __init__(self, delta=None, dim=64, activation=fn.tanh,
                 max_initial_norm=0.1, ntrunc:bool=False):
        """Initialize weights."""
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(dim, dim, bias=False)

        if ntrunc and abs(max_initial_norm) > 1e-9:
            nn.init.trunc_normal_(self.layer.weight, 
                                    mean = 0.0,
                                    std = max_initial_norm ** 0.5, 
                                    a   = - max_initial_norm, 
                                    b   = max_initial_norm) 
        else:
            nn.init.constant_(self.layer.weight, val=max_initial_norm)

    def forward(self, x, prev_block):
        """Forward pass."""
        update = prev_block.activation(prev_block.layer(x))
        return x + update @ (self.layer.weight - prev_block.layer.weight), self

class LinearSkip(nn.Module):
    def __init__(self, delta=None, dim=64, activation=fn.tanh,
                 max_initial_norm=0.1, ntrunc:bool=False):
        """Initialize weights."""
        super().__init__()
        self.activation = activation
        self.layer = nn.Linear(dim, dim, bias=False)

        if ntrunc and abs(max_initial_norm) > 1e-9:
            nn.init.trunc_normal_(self.layer.weight, 
                                    mean = 0.0,
                                    std = max_initial_norm ** (0.5), 
                                    a   = - max_initial_norm, 
                                    b   = max_initial_norm) 
        else:
            nn.init.constant_(self.layer.weight, val=max_initial_norm)

    def forward(self, x):
        """Forward pass."""
        return x + self.layer(self.activation(x))



class ResNet(nn.Module):
    """Fully connected ResNet."""

    def __init__(self, dim=64, L=10, in_features=1, 
                 out_features=1, delta=None, alpha=0.5,
                 gamma=0.5, activation=fn.tanh,
                 augmentation:str="skip", init:str="constant"):
        """Initialize blocks."""
        super().__init__()
        
        self.readin  = Reducer(in_features, dim)
        self.readout = Reducer(dim, out_features)

        # If delta is None treat it as a parameter, otherwise use the 
        # passed value.
        self.delta = delta if delta is not None else nn.Parameter(torch.Tensor([1.0]))
        self.alpha = alpha
        self.L     = L
        self.activation = activation
        self.augmentation = augmentation
        
        # max value to initialize the weights is defined by 
        # d^2 * L^{-1/gamma - beta} with beta = 1-alpha.
        w_scaling           = 1 / gamma + 1 - alpha
        max_initialization  = 1 / (L**w_scaling)
        sign                = lambda x : (1,-1)[x<0]
        if init == "constant":
            weights_scaling = [max_initialization/2 for i in range(L)]
        elif init == "ntrunc":
            weights_scaling = [max_initialization/2 for i in range(L)]
        elif init == "lin_dec":
            weights_scaling = [(L-i-1)** w_scaling * max_initialization**2 for i in range(L)]
        elif init == "lin_inc":
            weights_scaling = [i**w_scaling * max_initialization**2 for i in range(L)]
        elif init == "quad":
            weights_scaling = [((L-i-1)*i) ** w_scaling * max_initialization**3 for i in range(L)]
        elif init == "lin_dec_sym":
            weights_scaling = [(sign(L-2*(i-1)) / 2 ** w_scaling) *
                                    abs(L-2*(i-1)) ** w_scaling * 
                                        max_initialization**2  for i in range(L)]
        elif init == "lin_inc_sym":
            weights_scaling = [(sign(2*i - L) / 2 ** w_scaling) * 
                                abs(2*i-L) ** w_scaling * 
                                    max_initialization**2 for i in range(L)]
        else:
            raise AttributeErorr(f"The specified initialization '{init}' is not supported")

        ntrunc = init=="ntrunc"

        if augmentation == "skip":
            self.blocks = nn.Sequential(*[Block(self.delta, 
                                                dim, 
                                                activation,
                                                w, ntrunc=ntrunc) for w in weights_scaling])
        elif augmentation == "inc_skip":
            self.blocks = nn.Sequential(*[IncSkipBlock(self.delta, 
                                            dim, 
                                            activation,
                                            w, ntrunc=ntrunc) for w in weights_scaling])

            self.z = IncSkipBlock(0, dim, activation, 0.0)
        elif augmentation == "linear_skip": 
            self.blocks = nn.Sequential(*[LinearSkip(self.delta, 
                                            dim, 
                                            activation,
                                            w, ntrunc=ntrunc) for w in weights_scaling])
        elif augmentation == "none": 
            self.blocks = nn.Sequential(*[ActivationLinear(self.delta, 
                                            dim, 
                                            activation,
                                            w, ntrunc=ntrunc) for w in weights_scaling])
        else:            
            raise AttributeError(f"The specified augmentation '{augmentation}' is not supported.")

    def forward(self, x):
        """Forward pass."""
        if self.augmentation == "inc_skip":
            y = self.readin(x)
            z = self.z
            for layer in self.blocks:
                y, z = layer.forward(y, z)
            return self.readout(y)

        return self.readout(self.blocks(self.readin(x)))


