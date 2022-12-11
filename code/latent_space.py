import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load('../models/best_model_saved_at_15_percent_low_diff.pt', map_location=device))
model.eval()

convert_tensor = transforms.ToTensor()

img1 = convert_tensor(Image.open("../data/unittest/all/beachfamily0.png")).to(device)
img2 = convert_tensor(Image.open("../data/unittest/all/beachfamily7.png")).to(device)

# Encode and reparameterize to get the latent vector
z1 = model.reparameterize(*model.encode(img1.unsqueeze(0)))
z2 = model.reparameterize(*model.encode(img2.unsqueeze(0)))

# Interpolate between the two latent vectors
z_inter = torch.zeros(6, 64, 8, 8).to(device)
for i in range(6):
    z_inter[i] = (z1 + (z2 - z1) * i/5)

# Decode the interpolated latent vectors
interpol_latent = model.decode(z_inter)

# Create plot of the interpolation
fig, ax = plt.subplots(2, 6, figsize=(10, 10))
for i in range(6):
    ax[0, i].imshow(interpol_latent[i].cpu().permute(1, 2, 0).detach().numpy())
    ax[0, i].set_title(f'interpol latent{i}')

# Interpolate actual images and plot
interpol_imgs = torch.zeros(6, 3, 128, 128).to(device)
for i in range(6):
    interpol_imgs[i] = img1 + (img2 - img1) * i/5
    ax[1, i].imshow(interpol_imgs[i].cpu().permute(1, 2, 0).detach().numpy())
    ax[1, i].set_title(f'interpol img{i}')
plt.show()

# Decode a random latent vector
randomnoise = torch.randn(1, 64, 8, 8).to(device)
randomnoise = randomnoise.to(device)
randomnoiseimage = model.decode(randomnoise)
plt.imshow(randomnoiseimage[0].cpu().permute(1, 2, 0).detach().numpy())
plt.show()
