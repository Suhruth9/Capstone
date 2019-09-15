import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

dataset = torchvision.datasets.ImageNet('/storage/research/VQA/mnist', download = True)


train_dataset, val_dataset = random_split(dataset, [80, 20])


train_loader = DataLoader(train_dataset, batch_size =  )
val_loader = DataLoader(val_dataset, batch_size = )

model = pixelRNN()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

train_losses = []
val_losses = []

print(model.state_dict())

for epoch in range(200):
    batch_losses = []
    for x_batch, _ in train_loader:

        model.train()
        output = model(x_batch)
        loss = loss_fn(output, x_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss)

    loss = np.mean(batch_losses)
    train_loss.append(loss)

    with torch.no_grad():

        batch_losses = []
        for x_batch, _ in val_loader:
            model.eval()
            output = model(x_batch)
            val_loss = loss_fn(output, x_batch)
            batch_losses.append(val_loss)


        val_loss = np.mean(batch_losses)
        val_losses.append(val_loss)

    print(f"[{epoch+1}] Training loss: {loss:.3f}\t Validation loss: {val_loss:.3f}")
        









