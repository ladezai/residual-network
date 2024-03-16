import torch
import torch.nn as nn
import torch.utils.data as dt

import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib import cm

from resnet import ResNet, Block
from dataset import SinDataModule, CIFAR10DataModule
from ewfcm import load_model 

from typing import Optional, Union
from abc import abstractmethod

class ModelWrapper():
    def __init__(self, model : nn.Module):
        self.model = model
    
    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        """
            Returns the model's parameter as a tensor.
        """
        pass

    @abstractmethod
    def from_tensor(self, params : torch.Tensor):
        """
            Generates the model, modify the wrapped model in-place
        """
        pass

    def to(self, val):
        self.model = self.model.to(val)
        return self

    def forward(self, x):
        return self.model(x)


class Metric():
    def __init__(self, criterion : nn.Module, 
                 data : dt.DataLoader, 
                 dev : str = "cpu"):
        self.device = torch.device(dev)
        self.data = data 
        self.criterion = criterion.to(self.device)

    def __call__(self, model : Union[nn.Module, ModelWrapper]):
        # Computes the average loss given a model 
        # with respect to the pre-defined data and 
        # criterion
        avg_loss = 0.0
        data_len = 0
        for sample in self.data:
            #print(sample)
            inp, lab = sample
            inp, lab = inp.to(self.device), lab.to(self.device)
            outs     = model.to(self.device).forward(inp)
            loss_val = self.criterion(outs, lab)
            avg_loss+= loss_val.item()
            data_len+= 1

        return avg_loss / data_len

def tensor_dot(a : torch.Tensor, b : torch.Tensor, 
               dim : Optional[tuple[int]],
               dev : str = "cpu"):
    """
        Does the tensor product only on the first axis.   
    """
    assert a.shape == b.shape, f"{a.shape} != {b.shape}: same shape between tensors is required."
    
    device = torch.device(dev)
    return (a * b).sum(dim=dim).to(device)

def orthogonal_to(dir : torch.Tensor, 
                  dev : str = "cpu"):
    """
        Generates an almost orthogonal vector to the given direction. 
    """
    device = torch.device(dev)
    # generate a randomly sampled tensor
    new_vector = nrand_like(dir.shape, dev).to(device)
    norm_dir = (torch.linalg.norm(dir, dim=[1,2])**2).to(device)
    dot = tensor_dot(new_vector, dir, dim=(1,2),dev=dev)
    mat_dim = new_vector.shape[1]
    # Orthogonalize with respect to the given dir tensor 
    dot_over_norm = torch.stack(tuple(torch.ones(mat_dim, mat_dim).to(device) * dot[i] / norm_dir[i] for i in
                range(len(dot)))).to(device)
    new_vector -= dot_over_norm * dir
    # re-normalize
    norm_new_vector = torch.linalg.norm(new_vector, dim=[1,2]).to(device)
    normalization = torch.stack(tuple(torch.ones(mat_dim, mat_dim).to(device) *
        norm_new_vector[i] for i in range(len(dot)))).to(device)

    new_vector /= normalization
    return new_vector

def nrand_like(shape : torch.Size, dev : str = "cpu"):
    """
        Generates randomly sampled points by the second axis. 
        
        This works correctly for tensors of shape [n, d, d] with 
        n being the axes of the layer dimension, while d x d represents 
        the size of each layer.
    """
    device = torch.device(dev)
    loader = torch.stack(tuple([torch.normal(mean=0.0, 
                        std=torch.ones(shape[1:])).to(device) 
                        for i in range(shape[0])])).to(device)
    return loader

def normalize_filter(dir : torch.Tensor, theta : torch.Tensor, dev : str =
                     "cpu"):
    """
        Normalization as described in Visualizing the Loss Landscape of Neural
        Nets[1].

        [1] https://arxiv.org/abs/1712.09913v3
    """
    device  = torch.device(dev)
    scaling = torch.linalg.norm(theta, dim=[1,2]) / torch.linalg.norm(dir, dim=[1,2]) 
    mat_dim = dir.shape[1] 
    scaling_tensor = torch.stack(
        tuple(torch.ones(mat_dim, mat_dim).to(device) * scaling[i] 
            for i in range(len(scaling)))).to(device)
    
    dir *= scaling_tensor
    return dir

def parameters_to_tensor(model : nn.Module, dev : str = "cpu"):
    """
        Load the paramaters into a tensor.

        Each sub-module should gets on a new dimension
        of the 0 axis.

        :raise: ValueError if the model doesn't have parameters.
    """
    device = torch.device(dev)
    param_gen = model.parameters()
    tuple_param = tuple(param_gen)
    if len(tuple_param) == 0:
        raise ValueError(f"Not enough layers specified in module {model}") 
    elif len(tuple_param) == 1:
        # add a new axis
        loader = param_gen[0].reshape(1, *param_gen[0].shape)
    else:
        loader = torch.stack(tuple_param).to(device)
    return loader

def generate_filtered_gaussian_plane(n   : int, 
                                     res : float,
                                     model : ModelWrapper,
                                     metric : Metric,
                                     normalization : Optional[str] =  None,
                                     dev : str = "cpu"):
    """
        Samples via normal distribution n filtered planar directions 
        with respect to the given parameters theta.
        
        :n: int.
        :res: float, norm of each direction
        :model: ModelWrapper
        :metric: Metric
        :normalization: Optional[str] default = None.
        :dev: str, default = "cpu".
    """
    device = torch.device(dev)
    theta  = model.to_tensor().to(device)
    # Initial point
    point  = theta.detach()

    # generate a random direction the same shape as theta
    dir_one  = nrand_like(theta.shape, dev)
    dir_two  = orthogonal_to(dir_one, dev)    

    # normalize the direction with respect to the layers
    if normalization is not None:
        if normalization == "filter":
            dir_one = normalize_filter(dir_one, theta, dev)
            dir_two = normalize_filter(dir_two, theta, dev)

    dir_one *= res
    dir_two *= res
    # number of steps is defined by n
    data = torch.zeros(n, n)
    # set the initial point at the center
    point -= dir_one  * (n//2)  + dir_two * (n//2)
    # NOTE: this procedure could be parallelized, however Python 
    # is not that efficient (moreover the memory overhead could be 
    # quite large since a new model have to be stored for each different 
    # thread).
    for i in range(n):
        print(f"row: {i}/{n}\r")
        for j in range(n):
            model.from_tensor(point)
            data[i,j] = metric(model)
            point += dir_one 
            print(f"col: {j}/{n}\r")
        # reset the position with respect to the dir_one axis.
        point -= dir_one * n
        # move on the dir_two axis.
        point += dir_two

    return data

# TESTS
def ortho_test():
    print("""
        ## TEST on a
    """)
    a = torch.Tensor([[
        [0.5, 0.0], 
        [1.0, 0.1]]])
    #b = torch.Tensor([[1], [0], [0]))
    res = orthogonal_to(a)
    print(f"{res}\nDot product with a: {tensor_dot(res, a,dim=(1,2))}")

    print("""
        ## TEST on b
    """)
    b = torch.Tensor([
        [[0.0, 1.0],
         [1.0, 0.3]],
        [[0.5, 4.0], 
         [0.4, -0.01]]])
    res = orthogonal_to(b)
    corr = tensor_dot(res, b, dim=(1,2)) / (torch.linalg.norm(b, dim=[1,2]) *
                                            torch.linalg.norm(res, dim=[1,2]))
    print(f"{res}\nCorrelation with b: {corr}")


class ResnetWrapped(ModelWrapper):
    def __init__(self, model : nn.Module):
        self.model = model

    def to_tensor(self):
        try:
            blocks = self.model.get_submodule("blocks")
        except AttributeError:
            resnet = self.model.get_submodule("resnet")
            blocks = resnet.get_submodule("blocks")

        param_gen = blocks.parameters()
        tuple_param = tuple(param_gen)
        if len(tuple_param) == 0:
            raise ValueError(f"Not enough layers specified in module {model}") 
        elif len(tuple_param) == 1:
            # add a new axis
            loader = param_gen[0].reshape(1, *param_gen[0].shape)
        else:
            loader = torch.stack(tuple_param)
        return loader

    def from_tensor(self, params : torch.Tensor):
        layers = params.shape[0]

        try:
            blocks = self.model.get_submodule("blocks")
        except AttributeError:
            resnet = self.model.get_submodule("resnet")
            blocks = resnet.get_submodule("blocks")

        #assert params.shape[0] == len(blocks), "must have the same values of blocks"

        for i, block in enumerate(blocks):
            block.layer.weight.data = params[i,:]

def small_loss_landscape_visualization_test():
    """
        Loss landscape visualization test
    """
    L = 10
    model = ResNet(dim=2,L=L,delta=L**0.5,alpha=0.5)
    modelWrapper= ResnetWrapped(model)
     
    data = SinDataModule(num_workers=2, num_samples=5,c=0.01)
    data.prepare_data()
    data.setup("fit")
    dataloader = data.train_dataloader()
    metric = Metric(nn.L1Loss(), dataloader, dev="mps")
    n = 100 
    res = 0.1
    surface = generate_filtered_gaussian_plane(n,
                                     res,
                                     modelWrapper, 
                                     metric,
                                     normalization="filter",
                                     dev="mps")
    surface = surface.detach().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
    X = np.linspace(-1, 1, n)
    Y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X,Y, surface, cmap=cm.spring)
    plt.show()


def cifar10_landscape_filtered_ndir(config_path : str, 
                                    checkpoint_path : str, 
                                    steps : int = 100, 
                                    res : float = 0.1, 
                                    dev : str = "cpu"):
    model = load_model(config_path, checkpoint_path)
    model.eval()
    modelWrapper = ResnetWrapped(model)

    # Gets only 20 samples 
    data  = CIFAR10DataModule(training_data_percentage = 1/3_000,
                              num_workers=3)
    data.prepare_data()
    data.setup("fit")

    dataloader = data.train_dataloader()
    metric     = Metric(nn.CrossEntropyLoss(), dataloader, dev=dev)
    surface    = generate_filtered_gaussian_plane(steps, res, modelWrapper, metric,
                                               normalization="filter",
                                               dev=dev)
    return surface  

def store(surface : np.ndarray, out : str ="example"):
    """
        Store a surface into a .npy matrix. 
    """
    path = f"{out}.npy"
    if type(surface) == np.ndarray:
        np.save(path, surface)
    elif type(surface) == torch.Tensor:
        np.save(path, surface.detach().numpy())

def load_surf_and_visualize(path : str):
    """
        Plots both the surface and contour plot of a given 
        loss landscape.
    """
    # Load the surface with increments
    surface = np.load(path)
    X = np.linspace(-1, 1, surface.shape[0]) 
    Y = np.linspace(-1, 1, surface.shape[1])
    X, Y = np.meshgrid(X, Y)
    
    # Show the surface visualization
    fig, ax = plt.subplots(subplot_kw={"projection" : "3d"})
    ax.plot_surface(X, Y, surface, cmap='viridis')
    ax.set_title("")
    print(f"./{path.split('.')[0]}.png")
    plt.savefig(f"./{path.split('.')[0]}.png", dpi=300)
    plt.show()

    # Show contour plot
    plt.figure()
    plt.contour(surface, levels=50)
    plt.title('Loss Contour')
    plt.show()

if __name__ == "__main__":
    # Creates a parser for the visualize_landscape using CIFAR10 dataset
    # 
    # To visualize a surface, use --visualize=true --surface_path=example.npy
    #
    # To compute the loss landscape use 
    # --config_path=path/to/config.yaml \
    # --checkpoint_path=path/to/checkpoint.ckpt \
    # --steps=n \
    # --res=(1/L) * constant (L = Number of Layers) \
    # --device=cuda \
    # --out=surface_new_path
    parser = ArgumentParser(prog="Landscape Visualizer")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--steps", dest='steps', type=int, default=100)
    parser.add_argument("--res", dest='res', type=float, default=0.05)
    parser.add_argument("--seed", dest='seed', type=int, default=1234)
    parser.add_argument("--device", dest='device', type=str, default="cpu")
    parser.add_argument("--visualize", dest='visualize', type=bool,
                        default=False)
    parser.add_argument("--surface_path", dest='surf_path', type=str)
    parser.add_argument("--out", dest='out_path', type=str,
                        default='example')

    args = parser.parse_args()
    
    if not args.visualize:
        # Visualization 
        torch.manual_seed(args.seed)
        surf = cifar10_landscape_filtered_ndir(
                config_path=args.config_path, 
                checkpoint_path=args.checkpoint_path, 
                steps=args.steps, res=args.res, dev=args.device)
        surf = surf.detach().numpy()
        # Store the surface
        store(surf, args.out_path)
    else:
        load_surf_and_visualize(path=args.surf_path)
