import torch.nn as nn

# GAN network
class D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid() # binary classification
        )
        
    def forward(self, x):
        return self.discriminator(x)
    
class G(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.generator(x)


    