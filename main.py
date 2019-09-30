import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import scipy
#from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from PixelRNN import PixelRNN
import torchvision
import torchvision.transforms as transforms
import time

train_dataset = torchvision.datasets.MNIST('/storage/research/VQA/mnist', download = True, transform=transforms.Compose([transforms.ToTensor()]) )
val_dataset = torchvision.datasets.MNIST('/storage/research/VQA/mnist', train = False, download = True, transform=transforms.Compose([transforms.ToTensor()]) )



#rain_dataset, val_dataset = random_split(dataset, [50000, 10000])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_loader = DataLoader(train_dataset, batch_size =  16)
val_loader = DataLoader(val_dataset, batch_size = 64)

model = PixelRNN()
#model = nn.DataParallel(model)
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

train_losses = []
val_losses = []

print(model.state_dict())
print(len(model.state_dict()))
#print(model.state_dict()["output_conv2d.weights"].type())

print("OK1")
for epoch in range(200):
    batch_losses = []
    for x_batch, _ in train_loader:
        
        
        model.train()
        
        x_batch = x_batch.to(device)
        x_batch = Variable(x_batch)
    
        output = model(x_batch)
        loss = loss_fn(output, x_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
        
        
        
    loss = np.mean(batch_losses)
    train_losses.append(loss)

    with torch.no_grad():

        batch_losses = []
        for x_batch, _ in val_loader:
            model.eval()
            x_batch = x_batch.to(device)
            output = model(x_batch)
            val_loss = loss_fn(output, x_batch)
            batch_losses.append(val_loss.item())


        val_loss = np.mean(batch_losses)
        val_losses.append(val_loss)

    print(f"[{epoch+1}] Training loss: {loss:.3f}\t Validation loss: {val_loss:.3f}")
    if epoch%10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, "mnist_model_epoch"+str(epoch))
        








