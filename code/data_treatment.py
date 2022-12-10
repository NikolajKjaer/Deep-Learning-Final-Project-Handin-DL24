import torch
import torchvision
import os
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# print(os.getcwd())

OUTPUT_IMAGE_SIZE = 128

# Transforming images so smallest dimension is 128 pixles using the resize transform and then center crop to make 128x128
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(OUTPUT_IMAGE_SIZE),
    torchvision.transforms.CenterCrop(OUTPUT_IMAGE_SIZE),
    torchvision.transforms.ToTensor()
])


def square(image, mask=None):
    if mask is None:
        mask = torch.zeros((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    upper_left = torch.randint(0, OUTPUT_IMAGE_SIZE - OUTPUT_IMAGE_SIZE//7 - 1, (2,))
    square_shape = torch.randint(OUTPUT_IMAGE_SIZE//20, OUTPUT_IMAGE_SIZE//7, (2,))
    mask[:,upper_left[0]:upper_left[0] + square_shape[0], upper_left[1]:upper_left[1] + square_shape[1]] = 1
    # mask[upper_left[0]:upper_left[0] + square_shape[0], upper_left[1]:upper_left[1] + square_shape[1]] = random.random()
    noise = torch.rand((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    reduced_img = torch.zeros((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    reduced_img[torch.where(mask == 0)] = image[torch.where(mask == 0)]
    reduced_img[torch.where(mask == 1)] = noise[torch.where(mask == 1)]

    return reduced_img, mask

def circle(image, mask=None):
    if mask is None:
        mask = torch.zeros((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    center = torch.randint(OUTPUT_IMAGE_SIZE//15 + 1, OUTPUT_IMAGE_SIZE - OUTPUT_IMAGE_SIZE//15 - 1, (2,))
    radius = torch.randint(OUTPUT_IMAGE_SIZE//30, OUTPUT_IMAGE_SIZE//15, (1,))
    circle_pixels = torch.where((torch.arange(OUTPUT_IMAGE_SIZE).reshape(OUTPUT_IMAGE_SIZE, 1) - center[0]) ** 2 + (
            torch.arange(OUTPUT_IMAGE_SIZE).reshape(1, OUTPUT_IMAGE_SIZE) - center[1]) ** 2 <= radius ** 2)
    mask[:,circle_pixels[0], circle_pixels[1]] = 1
    noise = torch.rand((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    reduced_img = torch.zeros((3, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE))
    reduced_img[torch.where(mask == 0)] = image[torch.where(mask == 0)]
    reduced_img[torch.where(mask == 1)] = noise[torch.where(mask == 1)]

    return reduced_img, mask

# cum_mask = 0
#num_pictures = 10000
i = 0
for types in tqdm(['train', 'test', 'validation']):
    print(f'Currently working on {types} pictures.')
    images = torchvision.datasets.ImageFolder(root=f'../image_input/{types}/gt', transform=transform) 
    #https://github.com/CSAILVision/miniplaces
    for img_name, image in zip(images.imgs, images):
        img_name = img_name[0].replace(f'../image_input/{types}/gt/all', '')
        img_name = img_name.replace('/', '')
        img = image[0]
        mask = None
        img, mask = square(img, mask)
        for amount in [5, 10, 15]:
            for _ in range(amount):
                img, mask = square(img, mask)
                img, mask = circle(img, mask)
            final_img = torch.cat((img, mask[0, :, :].reshape(1, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)), dim=0)
            torchvision.utils.save_image(final_img, f"../image_input/{types}/{amount}/{img_name[:-4]}.png")

        # plt.imshow(mask)
        # plt.show()
        # cum_mask += torch.sum(mask)
        #final_img = torch.cat((img, mask.reshape(1, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)), dim=0)
        #print(img_name[:-4])
        #torchvision.utils.save_image(final_img, "../final_input_data/all/"+img_name[:-4]+".png")

# print(cum_mask / (OUTPUT_IMAGE_SIZE * OUTPUT_IMAGE_SIZE * len(images)))


# Used for reading images with 4 channels
def custom_pil_loader(path):  # https://github.com/pytorch/vision/issues/2276
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA")  # A is for alpha channel, but here it is just a mask

# Example:
testing_images = torchvision.datasets.ImageFolder(root='../final_input_data',
                                                  transform=torchvision.transforms.ToTensor(), 
                                                  loader=custom_pil_loader)

# plot the first 3 channels of the first image
plt.imshow(testing_images[0][0][0:3].permute(1, 2, 0))
plt.show()
