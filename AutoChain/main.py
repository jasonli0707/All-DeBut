import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from model import VGG_Like
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from generate_chain import generate_debut_chains
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
epochs = 200


def train_loop(model, criterion, optimizer, train_loader, epoch):
    model.train()
    for data, target in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()

def test_loop(model, criterion, test_loader):
    model.eval()
    val_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        val_loss += criterion(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target).cpu().sum()
    
    val_loss =  val_loss/len(test_loader)

    val_acc = 100*(correct.to(torch.float32) / len(test_loader.dataset))
  
    print('\nTest set: Test loss: {:.4f}, Test Accuracy: {}/{} ({:.5f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset), val_acc))
    return val_loss, val_acc


def train(name, model, opt='adam'):
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root='./data/mnist_data', 
                                    train=True, 
                                    transform=transforms)

    test_dataset = torchvision.datasets.MNIST(root='./data/mnist_data', 
                                    train=False, 
                                    transform=transforms) 

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if opt == 'adam':
        # for debut 
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
    else:
        # for baseline
        optimizer = optim.SGD(model.parameters(),
                            lr=0.05,
                            momentum=0.9,
                            weight_decay=5e-4)
                            
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    best_acc = 0
    for epoch in range(epochs):
        time1 = time.time()
        train_loss = train_loop(model, criterion, optimizer, train_loader, epoch)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, train loss {:.4f}'.format(epoch + 1, time2 - time1, train_loss))

        with torch.no_grad():
            _, test_acc = test_loop(model, criterion, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, f'./AutoChain/save/{name}.pth')
        print("best acc: ", best_acc.item())
        lr_scheduler.step()

def test(model_path, model):
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                    torchvision.transforms.ToTensor()])

    test_dataset = torchvision.datasets.MNIST(root='./data/mnist_data', 
                                    train=False, 
                                    transform=transforms) 

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    state_dict_path = model_path
    state_dict = torch.load(state_dict_path)
    print('epoch {}, best_acc {:.2f}'.format(state_dict['epoch'], state_dict['best_acc']))
    model.to(device)
    model.load_state_dict(state_dict['model'])
    criterion = nn.CrossEntropyLoss().to(device)
    _, test_acc = test_loop(model, criterion, test_loader)
    return test_acc

 
if __name__ == "__main__":
    # [c_out, k*c_in]
    cfg = [[64, 32*3*3], [128, 64*3*3], [256, 128*3*3]]
    shrinking_level = 5

    r_shapes = generate_debut_chains(cfg, type='m', shrinking_level=shrinking_level)
    debut_model  = VGG_Like(debut=True, R_shapes=r_shapes)
    print("debut num params: ", sum(p.numel() for p in debut_model.parameters()))
    name = 'debut_shrinkspeed_{}_{}'.format(shrinking_level, 4)
    train(name , debut_model)
