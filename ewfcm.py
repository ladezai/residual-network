"""
# Extract Weights From Checkpoint Model (ewfcm)

###  Usage: 
```
pipenv run python $FILE --logs_dir PATH/TO/LOGS --weights_dir PATH/TO/NEW/FOLDER
--verbose [false | true]
```

### Options description
--logs_dir: (default = ./MNIST) path to a directory containing lightning_logs
file, i.e.  
        MNIST/
          |
          +- model1/lightning_logs/version_NUMBER/config.yaml
          +- model2lightning_logs/version_NUMBER/config.yaml
          .
          .

--weights_dir: (default = ./mnist_weights) an empty directory that will contain
the weights of the models loaded from all sub-folders in --logs_dir option.
--verbose: (default = false) prints debug informations.

"""

import lightning as pl
import torch
import torch.nn as nn

from main import BaseModel, RegressionModel, ClassificationModel, MyCLI

from jsonargparse import ArgumentParser
import sys
import os
from itertools import takewhile

from typing import Optional, Any

def gather_configs_and_checkpoints(base_folder : str = "./CIFAR10",
                                   verbose:bool=False) -> dict[str, list[tuple[str,str]]]:
    """
        Given a folder 
        MNIST/
          |
          +- model1/lightning_logs/version_NUMBER/config.yaml
          +- model2lightning_logs/version_NUMBER/config.yaml
          .
          .
          .
        Finds each config.yaml file and its relative checkpoint model, 
        then returns a dictionary that stores as key the 'model
        classification' and as value a tuple containing the 
        full config.yaml path and the first checkpoints stored.
    """
    config_ckpt = {}
    for (dirpath, dirnames, filenames) in os.walk(base_folder):
        for filename in filenames:
            #print(filename)
            if filename == 'config.yaml':
                # Model generalities: If we have a string like
                # /path/modelclassnameNUMBER/config.yaml then the resulting
                # model_class will be stripped out of the number of the left and
                # store the class name as such.
                model_class = dirpath.lstrip(base_folder)
                dirs = model_class.split("/")
                if dirs[0] == '':
                    model_class = dirs[1]
                else:
                    model_class = dirs[0]
                number_to_remove = "".join(takewhile(lambda c : c.isdigit(),
                                                model_class[::-1]))[::-1]
                model_class = model_class.rstrip(number_to_remove)

                # get first checkpoint
                config_path = f"{dirpath}/{filename}"
                checkpoints_path = f"{dirpath}/checkpoints/"
                # Get the first checkpoint file in the directory
                checkpoints = os.listdir(checkpoints_path)
                checkpoints = [c for c in checkpoints if c.endswith("ckpt")]
                checkpoint  = checkpoints[0]
                # restore the full path
                checkpoint_path = checkpoints_path + checkpoint

                if model_class in config_ckpt: 
                    config_ckpt[model_class].append((config_path,
                                                     checkpoint_path))
                else:
                    config_ckpt[model_class] = [(config_path,
                                                 checkpoint_path)]

                if verbose:
                    print(model_class)
                    print(checkpoint)
                    print(config_path)
    if verbose:
        print(config_ckpt)

    return config_ckpt

def load_model(config_path : str, checkpoint_path : str) -> BaseModel:
    """
        Load the model configuration, the checkpoint and finally restore
        the trained model.
    """
    # Parser configuration for loading the parameters of the model correctly
    parser = ArgumentParser()
    parser.add_argument('--model', type=dict) # to ignore model
    parser.add_argument("--trainer", type=dict)  # to ignore trainer
    parser.add_argument("--seed-everything", type=int)  # to ignore trainer
    parser.add_argument("--ckpt_path", type=Any)
    parser.add_argument('--data', type=dict)
    config = parser.parse_path(config_path)
    string_to_classes = {"__main__.ClassificationModel" :
                         ClassificationModel, 
                         "__main__.RegressionModel" :
                         RegressionModel}
    
    # Once found out the model settings, create a new instance of the model
    model_class  = string_to_classes[config['model']['class_path']] 
    model = model_class(**config['model']['init_args'])
    
    # Load the checkpoints data for the relative model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        # the hidden layer has been changed from a simple linear 
        # to a block layer, to account for this, we need to change the 
        # name to the weights key.
        checkpoint['state_dict']['resnet.z.layer.weight'] = checkpoint['state_dict']['resnet.z.weight']
        _ = checkpoint['state_dict'].pop('resnet.z.weight', None)
        model.load_state_dict(checkpoint['state_dict'])
    return model

def store_model_weights(model : BaseModel, out_folder : str):
    """
        Store the model weights consistent with the 
        after_fit procedure in `main.py`.
    """
    # Actually saves the weights in the usual format
    resnet = model.get_submodule("resnet")
    delta_parameter = isinstance(resnet.delta, nn.Parameter)
    if delta_parameter:
        delta = model.get_parameter("resnet.delta")

    weights = [block.layer.weight.data for block in resnet.get_submodule("blocks")]

    if delta_parameter:
        torch.save(delta.data, f"{out_folder}/delta_{len(weights)}.pth")

    torch.save(weights, f"{out_folder}/resnet_{len(weights)}.pth")

def manual_load_weights_from_chekpoint(checkpoint_path : str = "./checkpoints/checkpoint.ckpt",
                 weights_dir : str = "./models"):
    """
        DISCLAIMER: this is deprecated but still useful in case of a single
        model to load.
    """
    # NOTE: L and d should be the same as the checkpoint loaded
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(checkpoint.keys())
    model = RegressionModel(L=2000, p_loss=2.0, p_var=2.0, lr=1e-3, alpha=1.0,
                 dim=10, gamma=1, activation='tanh')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Actually saves the weights in the usual format
    resnet = model.get_submodule("resnet")
    delta_parameter = isinstance(resnet.delta, nn.Parameter)
    if delta_parameter:
        delta = model.get_parameter("resnet.delta")

    weights = [block.layer.weight.data for block in resnet.get_submodule("blocks")]

    if delta_parameter:
        torch.save(delta.data, f"{weights_dir}/delta_{len(weights)}.pth")

    torch.save(weights, f"{weights_dir}/resnet_{len(weights)}.pth")


def test():
    load_model(config_path="./CIFAR10/base_model0/lightning_logs/version_411907/config.yaml",
               checkpoint_path="./CIFAR10/base_model0/lightning_logs/version_411907/checkpoints/epoch=179-step=180.ckpt")
    gather_configs_and_checkpoints(verbose=True)

if __name__ == "__main__":
    # Parser
    parser =  ArgumentParser(prog="EWFCM: Extract Weights From Checkpoint Models")
    parser.add_argument("--logs_dir", dest='logs_dir', type=str)
    parser.add_argument("--weights_dir", dest='weights_dir', type=str)
    parser.add_argument("--manual", dest="manual", type=bool)
    parser.add_argument("--verbose", type=bool, default=False) 
    args = parser.parse_args()

    # Actual routine: first walk through the given directory,
    # load the models then store it in common directories if names 
    # suggest so, e.g. base_model0, ..., base_model100, would end up 
    # in the same repository base_model/resnet1000.pth, ...,
    # base_model/resnet1100.pth (if the models have respectively 1000 and 1100 
    # layers).
    if args.manual:
        manual_load_weights_from_chekpoint(args.logs_dir, args.weights_dir)
    else:
        config_ckpts = gather_configs_and_checkpoints(
                                        base_folder=args.logs_dir,
                                        verbose=args.verbose)
        for model_class_name in config_ckpts:
            if args.verbose:
                print(f"{model_class_name=}")
            out_dir_path = f"{args.weights_dir.rstrip('/')}/{model_class_name}"  
            # Create the directory if non-existent
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)

            if args.verbose:
                print(out_dir_path)
            # Load and store the models weights
            for config_path, checkpoint_path in config_ckpts[model_class_name]:
                if args.verbose:
                    print(f"{config_path=};\n {checkpoint_path=}\n")
                model = load_model(config_path=config_path, 
                                   checkpoint_path=checkpoint_path)
                if args.verbose:
                    print(f"{out_dir_path=}")
                store_model_weights(model=model, out_folder=out_dir_path)
   
