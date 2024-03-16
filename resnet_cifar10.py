import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser

class Shortcut(nn.Module): 
    def __init__(self, planes): 
        super().__init__()
        self.planes = planes

    def forward(self, x):
        return nn.functional.pad(x[:, :, ::2, ::2], 
                     (0, 0, 0, 0, 
                     self.planes, self.planes), "constant", 0)

class ConvNormBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, delta=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.delta = delta
        if stride != 1 or in_planes != planes:
            self.shortcut = Shortcut(planes//4)
           
    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x))) * self.delta
        y = self.bn2(self.conv2(y)) * self.delta 
        y += self.shortcut(x) 
        y = nn.functional.relu(y)
        return y


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.num_blocks = num_blocks
        # Initial Conv + Batch 
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Deep ResNet architecture with 16, 32 and 64 filters progressively.
        self.blocks16filters = self._build_layer(block, 16, num_blocks[0], stride=1)
        for layer in self.blocks16filters:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

        self.blocks32filters = self._build_layer(block, 32, num_blocks[1], stride=2)
        for layer in self.blocks32filters:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

        self.blocks64filters = self._build_layer(block, 64, num_blocks[2], stride=2)
        for layer in self.blocks64filters:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)

        self.linear = nn.Linear(64, num_classes)
        nn.init.kaiming_normal_(self.linear.weight)
        
        
    def _build_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride, 
                                delta=num_blocks**(-0.75))] 
        # The new in_planes are the output of the first layer
        self.in_planes = planes
        layers.extend([block(planes, planes, 1, delta=num_blocks**(-0.75))
                         for _ in range(num_blocks-1)])
        return nn.Sequential(*layers)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.blocks16filters(y) 
        y = self.blocks32filters(y) 
        y = self.blocks64filters(y) 
        y = nn.functional.avg_pool2d(y, y.size()[3])
        y = torch.flatten(y, start_dim=1)
        y = self.linear(y)
        return y


# NOTE L = num_blocks * 6 + 2
def resnet56():
    return ResNet(ConvNormBlock, [9, 9, 9])

def resnet104():
    return ResNet(ConvNormBlock, [17, 17, 17])

def resnet224():
    return ResNet(ConvNormBlock, [37, 37, 37])

# TRAINING PROCEDURE
if __name__ == "__main__":
    import torch.optim as optim
    
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()


    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(32, padding=4),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load the datasets with the transform specified
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    parser = ArgumentParser(prog="Trains a ResNet for CIFAR10.")
    parser.add_argument("--L", dest='L', type=int)
    parser.add_argument("--seed", dest='seed', type=int, default=1231)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.L not in [56, 104, 224]:
        raise Error("No available model for the specified L")
    
    alpha = 0.75
    beta  = 0.25
    L = args.L
    if args.L == 56:
        net = resnet56().to(device)
    elif args.L == 104:
        net = resnet104().to(device)
    elif args.L == 224:
        net = resnet224().to(device)

    eta = L**(alpha - beta - 0.05)
    print(f"{alpha=}; {beta=}; {eta=}; {L=}.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=eta)
    writer    = SummaryWriter("resnet_events")
    steps     = 0
    EPOCHS    = 16
    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            
            # Add the batch grad loss norm
            yhat        = nn.Parameter(outputs.detach())
            local_loss  = criterion(yhat, labels)
            local_loss.backward()
            loss_grad_norm = torch.linalg.norm(yhat.grad, ord=2,dim=-1).pow(2).mean()
            writer.add_scalar("avg_grad_norm", loss_grad_norm, steps)
            steps += 1

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print the loss each 100 sgd steps. 
            if i % 100 == 100-1:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    print('Finished Training')
    writer.close()    

    # TEST 
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    top2_pred    = {classname: 0 for classname in classes}
    total_pred   = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images).to(device)
            _, predictions = torch.max(outputs, 1)
            _, predictions2 = torch.topk(outputs, 2, dim=1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                if label in predictions2:
                    top2_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    print("Top 1 accuracy: ", sum([v for k,v in correct_pred.items()])/sum([v for k, v in total_pred.items()]) * 100)
    print("Top 2 accuracy: ", sum([v for k,v in top2_pred.items()])/sum([v for k, v in total_pred.items()]) * 100)
    # print accuracy for each class
    print("=========== TOP 1 Classification Accuracy ======== ")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    print("=========== TOP 2 Classification Accuracy ======== ")
    for classname, correct_count in top2_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    # RESNET 56 alpha=0.75 EPOCHS=16; test accuracy = 82.03 (1-shot), 100 (3-shot).
    # RESNET 104 alpha=0.75 EPOCHS=16; test accuracy = 84.49 (1-shot), 100 (3-shot) .
    # RESNET 224 alpha=0.75 EPOCHS=16; test accuracy = 81.54 (1-shot), 100 (3-shot) .

