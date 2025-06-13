import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


# [x] クラスや関数に型アノテーションを追加
# [x] クラスや関数にdocstringを記載
# todo windowを並行して出せるようにしたい、
# todo wiodnowを出しながら、処理は完了とさせたい

# hyperparameters
input_dim = 784  # x dimension
hidden_dim = 200  # neurons in hidden layers
latent_dim = 20  # z dimension
epochs = 30
learning_rate = 3e-4
batch_size = 32


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Encoder network for the VAE.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mu and sigma for the latent space.
        """
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Decoder network for the VAE.

        Args:
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
        """
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            z (torch.Tensor): Input tensor from the latent space.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick to sample from the latent space.

    Args:
        mu (torch.Tensor): Mu values from the encoder.
        sigma (torch.Tensor): Sigma values from the encoder.

    Returns:
        torch.Tensor: Sampled latent vector.
    """
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Variational Autoencoder (VAE) class.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the VAE loss function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Computed loss.
        """
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = -torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)
        return (L1 + L2) / batch_size


# datasets
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten),  # falatten
    ]
)
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    # todo loss をprintしている旨を表示に追加、エポック数も出したい
    print(loss_avg)
    losses.append(loss_avg)

# plot losses
epochs = list(range(1, epochs + 1))
plt.plot(epochs, losses, marker="o", linestyle="-")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# todo 以下で loss カーブを表示
plt.show()


# generate new images
with torch.no_grad():
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)
    x = model.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28)

grid_img = torchvision.utils.make_grid(
    generated_images, nrow=8, padding=2, normalize=True
)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
# todo 以下で生成した画像を表示
plt.show()
