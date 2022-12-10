import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Used for reading images with 4 channels
def custom_pil_loader(path):  # https://github.com/pytorch/vision/issues/2276
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA")  # A is for alpha channel, but here it is just a mask

class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, input_data, output_data):
            self.input_data = input_data
            self.output_data = output_data

        def __len__(self):
            return len(self.input_data)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            input_img = self.input_data[idx][0]
            output_img = self.output_data[idx][0]

            return input_img, output_img

# Load VAE
model = VAE().to(device)

# training function
def train(epoch, train_loader, valid_loader, folder):
    global best_valid_loss

    model.train()
    train_loss = 0
    for input_img, output_img in train_loader:
        input_img = input_img.to(device)
        output_img = output_img.to(device)
        optimizer.zero_grad()
        recon_img, recon_img_mask, mu, logvar = model(input_img)
        loss = model.loss_function(recon_img, recon_img_mask, output_img, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for input_img, output_img in valid_loader:
            input_img = input_img.to(device)
            output_img = output_img.to(device)
            recon_img, recon_img_mask, mu, logvar = model(input_img)
            loss = model.loss_function(recon_img, recon_img_mask, output_img, mu, logvar)
            valid_loss += loss.item()
    
    # Print training and validation loss
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
          epoch, train_loss/len(train_loader), valid_loss/len(valid_loader)))
    
    # Save the model if the validation loss is the best we've seen so far.
    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), f'../models/CAN_OVERWRITE.pt')
        best_valid_loss = valid_loss

    return train_loss, valid_loss  # Purely for plotting purposes

# Prepare for training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
nEpoch = 500  # Was 500, then, 600 then 50 during training
all_loss = []
folders = iter(["5", "10", "15"])
training_output = datasets.ImageFolder(root='../images/train/gt',
                                       transform=transforms.ToTensor())
validation_output = datasets.ImageFolder(root='../images/validation/gt',
                                       transform=transforms.ToTensor())

for epoch in tqdm(range(nEpoch)):
    if epoch % (nEpoch//3) == 0:
        try:
            folder = next(folders)
            best_valid_loss = float('inf')
        except StopIteration:
            pass
    
        training_input = datasets.ImageFolder(root=f'../data/images/train/{folder}',
                                      transform=transforms.ToTensor(), 
                                      loader=custom_pil_loader)
        training_data = CustomDataset(training_input, training_output)
        training_loader = torch.utils.data.DataLoader(training_data, batch_size=16, shuffle=True)
                
        validation_input = datasets.ImageFolder(root=f'../data/images/validation/{folder}',
                                      transform=transforms.ToTensor(), 
                                      loader=custom_pil_loader)
        validation_data = CustomDataset(validation_input, validation_output)
        validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=16, shuffle=True)
            
    train_loss, valid_loss = train(epoch, training_loader, validation_loader, folder)
    all_loss.append((train_loss, valid_loss))


    # Save loss plot
    loss_for_plot = list(zip(*all_loss))
    plt.plot(np.arange(epoch+1), loss_for_plot[0])
    plt.plot(np.arange(epoch+1), loss_for_plot[1])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("../plots/lossplot.pdf")
