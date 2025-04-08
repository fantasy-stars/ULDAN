import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------#
# Martin Arjovsky, Soumith Chintala, and Léon Bottou. 2017. Wasserstein generative adversarial networks.
#------------------------------------------------------------------------------------------------------------#

latent_dim = 100
output_dim = 1024
epochs = 100000
n_critic = 5
lambda_gp = 10
batch_size = 64
 
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
 
    def forward(self, x):
        return self.net(x)
 
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
 
    def forward(self, x):
        return self.net(x)
 
generator = Generator(latent_dim, output_dim)
critic = Critic(output_dim)
 
opt_gen = optim.RMSprop(generator.parameters(), lr=0.00005)
opt_critic = optim.RMSprop(critic.parameters(), lr=0.00005)
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
 
train_dataset = datasets.MNIST(root='', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
def get_real_data(batch_size):
    for data in train_loader:
        real_data, _ = data
        real_data = real_data.view(real_data.size(0), -1)
        return real_data
 
def gradient_penalty(critic, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1).to(real_data.device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)
 
    critic_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).to(real_data.device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
 
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
 
 
for epoch in range(epochs):
    for _ in range(n_critic):
        opt_critic.zero_grad()
 
        real_data = get_real_data(batch_size)
        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
 
        loss_critic = torch.mean(critic(fake_data)) - torch.mean(critic(real_data))
 
        gp = gradient_penalty(critic, real_data, fake_data)
 
        loss_critic += lambda_gp * gp
 
        loss_critic.backward()
        opt_critic.step()
 
    opt_gen.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    fake_data = generator(z)
    loss_gen = -torch.mean(critic(fake_data))
    loss_gen.backward()
    opt_gen.step()
 
 
z = torch.randn(1, latent_dim)
generated_image = generator(z).view(32, 32).detach().numpy()
plt.imshow(generated_image, cmap='gray')
plt.show()

