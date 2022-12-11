import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
class enc_ResNet(nn.Module):
    def __init__(self):
        super(enc_ResNet, self).__init__()
        self.resnet = resnet
        self.resnet.eval()
        self.resnet.requires_grad_(False)
        
    def forward(self, x, num_layers):
        for i, layer_name in enumerate(self.resnet._modules.keys()):
            if i == num_layers:
                break
            
            x = self.resnet._modules[layer_name](x)
        
        return x
    
    def gram(self, features):
        (b, ch, h, w) = features.size()
        features = features.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = torch.bmm(features, features_t) / (ch * h * w)
        return gram
    
    def featureStyle_loss(self, x1, x2, num_layers=10):
        loss = 0
        for i in range(1, num_layers):
            x1_feat = self.forward(x1, i)
            x2_feat = self.forward(x2, i)
            loss += F.mse_loss(x1_feat, x2_feat)
            loss += F.mse_loss(self.gram(x1_feat), self.gram(x2_feat))

        return loss


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # encoder
        self.conv0 = nn.Conv2d(3, 5, 3, 1, 1)  # input: 3x128x128, output: 5x128x128
        self.batch0 = nn.BatchNorm2d(5)
        self.conv1 = nn.Conv2d(5, 8, 3, 2, 1)  # input: 5x128x128, output: 8x64x64
        self.batch1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 2, 1)  # input: 8x64x64, output: 16x32x32
        self.batch2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, 2, 1)   # input: 16x32x32, output: 32x16x16
        self.batch3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, 2, 1)  # input: 32x16x16, output: 64x8x8
        self.batch4 = nn.BatchNorm2d(64)
        self.convMu = nn.Conv2d(64, 64, 3, 1, 1)  # input: 64x8x8, output: 64x8x8
        self.convLogVar = nn.Conv2d(64, 64, 3, 1, 1)  # input: 64x8x8, output: 64x8x8

        # decoder
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)  # input: 64x8x8, output: 32x16x16
        self.batchd1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)  # input: 32x16x16, output: 16x32x32
        self.batchd2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)  # input: 16x32x32, output: 8x64x64
        self.batchd3 = nn.BatchNorm2d(8)
        self.deconv4 = nn.ConvTranspose2d(8, 5, 3, 2, 1, 1)  # input: 8x64x64, output: 5x128x128
        self.batchd4 = nn.BatchNorm2d(5)
        self.deconv5 = nn.Conv2d(5, 3, 3, 1, 1)  # input: 5x128x128, output: 3x128x128

    # reparaterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # encoder
    def encode(self, x):
        x = F.relu(self.conv0(x))
        x = self.batch0(x)
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        mu = self.convMu(x)
        logvar = self.convLogVar(x)
        return mu, logvar
    
    # decoder
    def decode(self, z):
        z = F.relu(self.deconv1(z))
        z = self.batchd1(z)
        z = F.relu(self.deconv2(z))
        z = self.batchd2(z)
        z = F.relu(self.deconv3(z))
        z = self.batchd3(z)
        z = F.relu(self.deconv4(z))
        z = self.batchd4(z)
        z = self.deconv5(z)
        return torch.sigmoid(z)
    
    # forward pass
    def forward(self, x):
        if x.shape[1] == 3:
            img = x
            mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(device)
        else:
            img, mask = x[:, :3, :, :], x[:, 3:, :, :]
        mu, logvar = self.encode(img)
        z = self.reparameterize(mu, logvar)
        z = self.decode(z)
        z_mask = z * mask + img * (1 - mask)
        return z, z_mask, mu, logvar
    
    # loss function
    def loss_function(self, recon_x, recon_x_mask, x, mu, logvar):
        # Perceptual and style loss
        featureStyle_loss = enc_resnet.featureStyle_loss(recon_x, x)
        featureStyle_loss += enc_resnet.featureStyle_loss(recon_x_mask, x)
        # difference of neighboring pixels (neighbour loss in report)
        diff_loss = F.mse_loss(recon_x_mask[:, :, :, :-1], recon_x_mask[:, :, :, 1:], reduction='sum')
        diff_loss += F.mse_loss(recon_x_mask[:, :, :-1, :], recon_x_mask[:, :, 1:, :], reduction='sum')
        # pixel loss
        pixel_loss = F.mse_loss(recon_x_mask, x, reduction='sum')
        pixel_loss += F.mse_loss(recon_x, x, reduction='sum')
        # KL divergence
        kldiv = torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar/2)
        
        # return pixel_loss*0.1 + diff_loss*10 + featureStyle_loss*100 + kldiv*0.01  # Used for the first 500 epochs
        # return pixel_loss*0.1 + diff_loss + featureStyle_loss*100 + kldiv*0.01  # Used for the next 600 epochs
        return pixel_loss*0.1 + diff_loss*0.01 + featureStyle_loss*100 + kldiv*0.01  # Used for the last 50 epochs

resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
enc_resnet = enc_ResNet().to(device)  # Used for loss function

if __name__ == "__main__":
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

    unittest_input = datasets.ImageFolder(root='../data/unittest',
                                      transform=transforms.ToTensor())
    unittest_dataset = CustomDataset(unittest_input, unittest_input)
    unittest_dataloader = torch.utils.data.DataLoader(unittest_dataset, batch_size=1, shuffle=True)
    unittest_iter = iter(unittest_dataloader)

    # Load model for unit test
    model = VAE().to(device)
    input_img, _ = next(unittest_iter)
    input_img = input_img.to(device)
    model(input_img)
