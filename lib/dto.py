import torch
import torch.optim as optim
from dataclasses import dataclass, field


@dataclass
class LossErrors:
    err_discriminator_x: list = field(default_factory=list)
    err_discriminator_z: list = field(default_factory=list)
    err_generator: list = field(default_factory=list)
    gp_history: list = field(default_factory=list)


@dataclass
class Optimizer:
    generator: optim.Adam
    discriminator: optim.Adam


@dataclass
class ImageInfoDTO:
    batch_size: int
    real_images: torch.Tensor
    labels: torch.Tensor


@dataclass
class TrainDiscriminatorOutDTO:
    err_discriminator_real: torch.Tensor
    err_discriminator_fake: torch.Tensor
    fake_images: torch.Tensor
    gen_labels: torch.Tensor
    gradient_penalty: torch.Tensor | None = None


@dataclass
class TrainGeneratorOutDTO:
    err_generator: torch.Tensor
