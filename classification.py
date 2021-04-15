import argparse
from pathlib import Path


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from logger import WandbLogger



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 125 * 125)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def save_checkpoint(net, optimizer, save_path):
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

def train(opt):
    wandb_logger = WandbLogger(project=opt.project)
    save_path = Path(opt.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if opt.log_dataset:
        wandb_logger.log_artifact(opt.dataset, "flower_data", "dataset")

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16

    trainset = torchvision.datasets.ImageFolder(root=opt.dataset+'/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root=opt.dataset+'/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('daisy', 'dandelion')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    latest_model_path =  save_path / 'latest.pt'
    best_model_path = save_path / 'best.pt'
    min_loss = len(trainloader)

    for epoch in range(opt.epochs):

        running_loss, total_loss = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 10 == 0:   
                wandb_logger.log({"loss": running_loss})
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        if total_loss < min_loss:
            min_loss = total_loss
        
        # Log Loss and checkpoint artifacts
        wandb_logger.log({"total_loss": total_loss})
        if (epoch+1) % opt.log_period == 0:
            save_checkpoint(net, optimizer, latest_model_path)
            wandb_logger.log_artifact(
                latest_model_path,
                "checkpoint"+ wandb_logger.run.id,
                "model",
                ['latest', 'epoch-'+str(epoch), 'best' if min_loss==total_loss else '']
            )
            print("logged model artifact after epoch ", epoch)
        wandb_logger.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="flower", help='dataset path')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--log_dataset', action='store_true', help='Log dataset as artifact')
    parser.add_argument('--log_period', type=int, default=5, help='set the checkpoint log period')
    parser.add_argument('--project', type=str, default="flower_classification", help='W&B project name')
    parser.add_argument('--save_path', type=str, default="checkpoint", help='path to save checkpoint')

    opt = parser.parse_args()
    
    train(opt)