import torch
from dataclasses import dataclass


@dataclass
class Config:
    batch_size = 64
    num_workers = 4
    num_epochs = 20
    noise_size = 100
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    log_image_every = 100
    latent_dim = 100
    image_shape = (1, 28, 28)
    img_channels = 1
    embed_size = 10
    num_classes = 10
    img_dim = 28 * 28
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_clip = 0.01  # Для WGAN с клиппингом весов
    lambda_gp = 10       # Коэффициент для штрафа градиентов в WGAN-GP
