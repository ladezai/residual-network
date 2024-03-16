"""Training and evualuation of model."""

import lightning as pl
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from lightning.pytorch.cli import LightningCLI

from dataset import SinDataModule, MNISTDataModule, CIFAR10DataModule, ConstantDataModule
from pvar import p_var
from resnet import ResNet, Reducer

from typing import Optional
import os

class BaseModel(pl.LightningModule):
    """Trainable model."""

    def __init__(self, 
                 L:int=10, 
                 dim:int=1, 
                 in_features:int=1,
                 out_features:int=1, 
                 p_loss:float=2.0,
                 p_var:float=2.0, 
                 lr:float=1e-3, 
                 alpha:float=0.5,
                 gamma:float=0.5, 
                 activation:Optional[str]=None,
                 augmentation:str="skip", 
                 init:str="constant", 
                 dynamic_lr:bool=False,
                 store_grad:bool=False):
        """Initialize weights."""
        super().__init__()
        
        if activation is None:
            activation = fn.tanh
        else:
            # load an activation function from a string. 
            activations = {'tanh'    : fn.tanh, 
                           'relu'    : fn.relu, 
                           'elu'     : fn.elu,
                           'gelu'    : fn.gelu, 
                           'sigmoid' : fn.sigmoid}
            activation = activations[activation] 

        self.dynamic_lr = dynamic_lr
        self.resnet = ResNet(
                dim=dim, 
                in_features=in_features, 
                out_features=out_features, 
                L=L, 
                delta=L ** (-alpha),
                alpha=alpha, 
                activation=activation, 
                gamma=gamma,
                augmentation=augmentation,
                init=init)
        self.flatten = nn.Flatten(start_dim=1)

        self.p_var = p_var
        self.p_loss = p_loss
        self.lr = lr
        self.store_grad = store_grad

    def forward(self, x):
        return self.resnet(self.flatten(x))
 
    def on_train_epoch_end(self):
        resnet = self.get_submodule("resnet")
        blocks = resnet.get_submodule("blocks")
        # rescale the weights appropriately 
        weights = [resnet.L**(1-resnet.alpha) * 
                    layer.layer.weight.data.cpu().detach().numpy().reshape(-1) 
                        for layer in blocks]

        # Compute the p-variation for the self.p_var, half its value and 3/4 of
        # its value.
        pv   = p_var(weights, self.p_var, "euclidean") ** (1 / self.p_var)
        pvH  = p_var(weights, self.p_var/2, "euclidean") ** (2 / self.p_var)
        pv2T = p_var(weights, self.p_var*3/4, "euclidean") ** (4 / (3 * self.p_var))
        
        self.log(f"{self.p_var:.3f}-variation", pv, prog_bar=True)
        self.log(f"{self.p_var/2:.3f}-variation", pvH, prog_bar=True)
        self.log(f"{self.p_var*3/4:.3f}-variation", pv2T, prog_bar=True)

        if self.dynamic_lr:
            # generally better accuracy with this change of values.
            c = 10.0 if self.resnet.activation == fn.relu else 2.0
            for g in self.optimizer.param_groups:
                g['lr'] = c * self.lr / (pv**2)
            self.log(f"lr", c * self.lr / (pv **2), prog_bar=True) 


    def configure_optimizers(self):
        """Configure optimizer."""
        # Do not update the embedding of the data of the resnet,
        # updates only the blocks. 
        resnet    = self.get_submodule("resnet")
        blocks    = resnet.get_submodule("blocks")
        self.optimizer = optim.SGD(blocks.parameters(), lr=self.lr)
        
        return self.optimizer


class RegressionModel(BaseModel):
    """Regression 1-to-1 trainable model."""

    def __init__(self, L:int=10, 
                 p_loss:float=2.0, 
                 p_var:float=2.0,
                 lr:float=1e-3, 
                 alpha:float=0.5,
                 gamma:float=0.5, 
                 dim:int=1,
                 activation:Optional[str]=None,
                 augmentation:str="skip", 
                 init:str="constant", 
                 dynamic_lr:bool=False,
                 store_grad:bool=False):
        """Initialize weights."""
        super().__init__(L=L, 
                         in_features=1, 
                         out_features=1, 
                         p_loss=p_loss,
                         p_var=p_var, 
                         lr=lr, 
                         alpha=alpha, 
                         gamma=gamma,
                         dim=dim,
                         activation=activation,
                         augmentation=augmentation, 
                         init=init,
                         dynamic_lr=dynamic_lr,
                         store_grad=store_grad)
 
    def forward(self, x):
        return super().forward(x) 
       
    def training_step(self, batch, batch_idx):
        """Predict and compute Cross Entropy loss."""
        x, y = batch
        pred = self.forward(x) 
        loss = torch.linalg.norm(y-pred, ord=self.p_loss, dim=-1).pow(self.p_loss).mean()
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        if self.store_grad: 
            yhat       = nn.Parameter(pred.detach())
            local_loss = torch.linalg.norm(y-yhat, ord=self.p_loss, dim=-1).pow(self.p_loss).mean()
            local_loss.backward()
            loss_grad_norm = torch.linalg.norm(yhat.grad,
                                               ord=2,dim=-1).pow(2).mean()
            self.log("avg_grad_norm", loss_grad_norm, on_epoch=True, on_step=False, prog_bar=True)
        return loss

class ClassificationModel(BaseModel):
    """MNIST/CIFAR10 Classification trainable model with 28*28 input data and a 10-dimensional output."""

    def __init__(self, L:int=10, p_loss:float=2.0, p_var:float=2.0,
                 lr:float=1e-3, alpha:float=0.5,
                 dim:int=1, gamma:float=0.5, classes:int=10, input_size:int=28*28,
                 activation:Optional[str]=None, 
                 augmentation:str="skip", 
                 init:str="constant", 
                 dynamic_lr:bool=False,
                 store_grad=False):
        """Initialize weights."""
        super().__init__(L=L, 
                         in_features=input_size,
                         out_features=classes, 
                         p_loss=p_loss,
                         p_var=p_var, 
                         lr=lr, 
                         alpha=alpha, 
                         gamma=gamma,
                         dim=dim,
                         activation=activation,
                         augmentation=augmentation,
                         init=init,
                         dynamic_lr=dynamic_lr,
                         store_grad=store_grad)
        # Setup metrics
        self.fit_metrics = nn.ModuleDict({
            "accuracy"  : MulticlassAccuracy(num_classes=classes),
            "precision" : MulticlassPrecision(num_classes=classes),
            "recall"    : MulticlassRecall(num_classes=classes)
        })
        # Setup Loss
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return super().forward(x) 

    def _shared_eval_step(self, batch, batch_idx):
        x, y  = batch
        y_hat = self.forward(x) 
        loss  = self.criterion(y_hat, y)
        return loss, y_hat
    
    def _metrics(self, pred, targets): 
        for name in self.fit_metrics:
            val = self.fit_metrics[name](pred, targets)
            self.log(name, val, on_epoch=True, on_step=False, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """Predict and compute Cross Entropy loss."""
        x, y = batch
        loss, pred = self._shared_eval_step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self._metrics(pred, y)
        if self.store_grad: 
            yhat        = nn.Parameter(pred.detach())
            local_loss  = self.criterion(yhat, y)
            local_loss.backward()
            loss_grad_norm = torch.linalg.norm(yhat.grad, ord=2,dim=-1).pow(2).mean()
            self.log("avg_grad_norm", loss_grad_norm, on_epoch=True, on_step=False, prog_bar=True)
 
        return loss
       
    def test_step(self, batch, batch_idx):
        """Test the models metrics (namely accuracy, precision and recall)"""
        x, y = batch
        loss, pred = self._shared_eval_step(batch, batch_idx)
        self._metrics(pred, y)
        

class MyCLI(LightningCLI):
    """Custom CLI that saves models."""

    def after_fit(self):
        """Save fitted model."""

        # If the folder exported_models does not exists it creates a new one.
        if not os.path.exists("./exported_models/"):
            os.makedirs("./exported_models/")

        # decide which are the parameter to store
        resnet = self.model.get_submodule("resnet")
        delta_parameter = isinstance(resnet.delta, nn.Parameter)
        if delta_parameter:
            delta = self.model.get_parameter("resnet.delta")
        
        weights = [block.layer.weight.data for block in resnet.get_submodule("blocks")]
    
        # store parameters
        if delta_parameter:
            torch.save(delta.data, f"./exported_models/delta_{len(weights)}.pth")

        torch.save(weights, f"./exported_models/resnet_{len(weights)}.pth")


def cli_main():
    """Main CLI interface loop."""
    MyCLI()


if __name__ == "__main__":
    import torch

    torch.set_float32_matmul_precision("medium")
    cli_main()
