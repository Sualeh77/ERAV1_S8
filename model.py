import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from utils import GetCorrectPredCount

########################################################################################################################################################
########################################################################  S6 | Mnist  #########################################################################
class Mnist_S6(nn.Module):
    """
     Model for Mnist under 20k params.
    """

    def __init__(self, device):
        super(Mnist_S6, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),                                                # 1x28x28 > 16x28x28 | RF 3
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),                                               # 16x28x28 > 16x28x28 | RF 5
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),                                               # 16x28x28 > 16x28x28 | RF 7
            nn.ReLU()
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                                                         # 16x28x28 > 16x14x14 | RF 8 | J 2
            #nn.Conv2d(32, 16, kernel_size=1)  # Removed as it was reducing acc from 99.59 to 99.53     # 32x14x14 > 16x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x14x14 > 16x12x12 | RF 12
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x12x12 > 16x10x10 | RF 16
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x10x10 > 16x8x8 | RF 20
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),                                               # 16x8x8 > 16x6x6 | RF 24
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),                                               # 16x6x6 > 32x4x4 | RF 28
        )    
        self.antman = nn.Conv2d(32, 10, 1)                                                # 32x4x4 > 10x4x4
        self.gap = nn.AvgPool2d(4)
        

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.antman(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

########################################################################  S7 | Mnist  #########################################################################
class Mnist_S7(nn.Module):
    """
        Model for Mnist under 8k params.
    """
    def __init__(self, device):
        super(Mnist_S7, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0),                  # 1x28x28 > 10x26x26 | 3
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                  # 10x26x26 > 10x24x24 | 5
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, 3, padding=0),                  # 10x24x24 > 20x22x22 | 7
            nn.ReLU(),
            nn.BatchNorm2d(20)
        )
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),                                # 20x22x22 > 20x11x11 | 7
            nn.Conv2d(20, 10, kernel_size=1),                   # 20x11x11 > 10x11x11 |7 | j = 2
            nn.BatchNorm2d(10)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x11x11 > 10x9x9 | 11
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                     # 10x9x9 > 10x7x7 | 15
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, 3, padding=0),                    # 10x7x7 > 10x5x5 | 19
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, 3, padding=0)                    # 10x5x5 > 20x3x3 | 23
        )
        #Stopping here as it seems 23 RF must cover number present in image. Leaving some room for Batch Norm params.
        self.op = nn.Sequential(
            nn.Conv2d(20, 10, kernel_size=1),               # 20x3x3 > 10x3x3
            nn.AdaptiveAvgPool2d((1,1))                                  # 10x3x3 > 10x1x1
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.op(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

########################################################################  S8 | Cifar 10   #########################################################################
class Cifar10(nn.Module):
    def __init__(self, device, norm:str="bn"):
        super(Cifar10, self).__init__()
        self.device = device
        self.norm = norm

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),     # 3x32x32 > 16x32x32 | RF 3
            nn.ReLU(),
            nn.GroupNorm(4, 16) if self.norm=="gn" else nn.GroupNorm(1, 16) if self.norm=="ln" else nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, 3, padding=1),        # 16x32x32 > 32x32x32 | RF 5
            nn.ReLU(),
            nn.GroupNorm(4, 32) if self.norm=="gn" else nn.GroupNorm(1, 32) if self.norm=="ln" else nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        )
        self.transition1 = nn.Sequential(
            nn.Conv2d(32, 16, 1),        # 32x32x32 > 16x32x32 | RF 5
            nn.MaxPool2d(2, 2)           # 16x32x32 > 16x16x16 | RF 6 | J 2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),        # 16x16x16 > 16x16x16 | RF 10
            nn.ReLU(),
            nn.GroupNorm(4, 16) if self.norm=="gn" else nn.GroupNorm(1, 16) if self.norm=="ln" else nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, 3, padding=1),       # 16x16x16 > 32x16x16 | RF 14
            nn.ReLU(),
            nn.GroupNorm(4, 32) if self.norm=="gn" else nn.GroupNorm(1, 32) if self.norm=="ln" else nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3, padding=1),      # 32x16x16 > 32x16x16 | RF 18
            nn.ReLU(),
            nn.GroupNorm(4, 32) if self.norm=="gn" else nn.GroupNorm(1, 32) if self.norm=="ln" else nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        )
        self.transition2 = nn.Sequential(
            nn.Conv2d(32, 16, 1),       # 32x16x16 > 16x16x16 | RF 18
            nn.MaxPool2d(2, 2)           # 16x16x16 > 16x8x8 | RF 20 | J 4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),        # 16x8x8 > 32x8x8 | RF 28
            nn.ReLU(),
            nn.GroupNorm(4, 32) if self.norm=="gn" else nn.GroupNorm(1, 32) if self.norm=="ln" else nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3),       # 32x8x8 > 32x6x6 | RF 36
            nn.ReLU(),
            nn.GroupNorm(4, 32) if self.norm=="gn" else nn.GroupNorm(1, 32) if self.norm=="ln" else nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3),      # 32x6x6 > 32x4x4 | RF 44
        )
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),    # 32x4x4 > 32x1x1
            nn.Conv2d(32, 10, 1)           # 32x1x1 > 10x1x1
        )

        
    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

##########################################################################################################################################################

########################################################################################################################################################

def model_summary(model, input_size=(1, 28, 28)):
    return summary(model, input_size)

########################################################################################################################################################


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


########################################################################################################################################################
def model_train(model, device, train_loader, optimizer, criterion, path):
    """
        Training method
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate Loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    torch.save(model.state_dict(), path)
########################################################################################################################################################


########################################################################################################################################################
def model_test(model, device, test_loader, criterion):
    """
        Test method.
    """
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            
            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#########################################################################################################################################################