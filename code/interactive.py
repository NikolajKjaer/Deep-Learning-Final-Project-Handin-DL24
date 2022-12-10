import tkinter as tk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from model import VAE
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
model.load_state_dict(torch.load("../models/best_model_saved_at_15_percent_low_diff.pt", map_location=device))
model.eval()

# Load image
img = Image.open("../data/interactive/Galaxy.png")
# img = Image.open("../data/interactive/StarryNight.jpg")
# img = Image.open("../data/interactive/Drawing.png")
img = img.resize((128, 128))  # Works best with size 128x128
width, height = img.size

# Create GUI where you can draw on an image
root = tk.Tk()
root.title("Draw on image")
root.geometry("{}x{}".format(width, height+50))
root.resizable(False, False)

# Create canvas
canvas = tk.Canvas(root, width=width, height=height, bg="white")
canvas.pack()

# Create image
image = ImageTk.PhotoImage(img)
canvas.create_image((width//2,height//2), image=image, state="normal")

# Create drawing function
def paint(event):
    global mask
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_rectangle((x1, y1, x2, y2), fill="black", width=brush_width)
    mask[:, event.y-brush_width//2-1:event.y+brush_width//2+1, event.x-brush_width//2-1:event.x+brush_width//2+1] = 1  # Square


# Bind mouse movement to drawing function
canvas.bind("<B1-Motion>", paint)

# Create brush size slider
brush_width = 5
brush_slider = tk.Scale(root, from_=1, to=25, orient=tk.HORIZONTAL, length=width, command=lambda x: change_brush_size(x))
brush_slider.set(brush_width)
brush_slider.pack()

# Create brush size change function
def change_brush_size(new_size):
    global brush_width
    brush_width = int(new_size)

# Create mask
mask = torch.zeros((3, height, width))

# Start GUI
root.mainloop()

# Create masked image
convert_tensor = transforms.ToTensor()
img = convert_tensor(img)
noise = torch.rand((3, height, width))
reduced_img = torch.zeros((3, height, width))
reduced_img[torch.where(mask == 0)] = img[torch.where(mask == 0)]
reduced_img[torch.where(mask == 1)] = noise[torch.where(mask == 1)]
img_w_mask = torch.cat((reduced_img, mask), dim=0)

recon_img, recon_img_mask, _, _ = model(img_w_mask.unsqueeze(0).to(device))

# Show reconstructed image
recon_img_np = recon_img.squeeze(0).permute(1,2,0).cpu().detach().numpy()

# Show reconstructed image with mask
recon_img_mask_np = recon_img_mask.squeeze(0).permute(1,2,0).cpu().detach().numpy()

# Make plot
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0, 0].imshow(img.permute(1,2,0))
ax[0, 0].set_title("Original image")
ax[0, 1].imshow(recon_img_mask_np)
ax[0, 1].set_title("Reconstructed image with mask")
ax[1, 1].imshow(recon_img_np)
ax[1, 1].set_title("Reconstructed image")
true_img_model = model(img.unsqueeze(0).to(device))[0]
ax[1, 0].imshow(true_img_model.squeeze(0).permute(1,2,0).cpu().detach().numpy())
ax[1, 0].set_title("True image after model")
ax[0, 2].imshow(reduced_img.permute(1,2,0))
ax[0, 2].set_title("Masked image")
ax[1, 2].imshow(mask.permute(1,2,0))
ax[1, 2].set_title("Mask")
plt.show()