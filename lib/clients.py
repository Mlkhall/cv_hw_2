import torch
from typing import Protocol
import plotly.graph_objs as go
import torchvision
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from torchvision.transforms import Compose, ToTensor, Normalize

from lib.dto import Optimizer, TrainDiscriminatorOutDTO, TrainGeneratorOutDTO, ImageInfoDTO, LossErrors
from lib.config import Config
from lib.modules import Generator, Discriminator
from plotly.subplots import make_subplots
import torchvision.utils as vutils
from torch.utils.data import DataLoader


class WGANTrainerProtocol(Protocol):

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        config: Config,
        writer: SummaryWriter,
        dataloader: DataLoader,
    ):
        self._generator = generator
        self._discriminator = discriminator
        self._writer = writer
        self._config = config
        self._dataloader = dataloader

    @property
    def fixed_noise(self) -> torch.Tensor:
        return torch.randn(64, self._config.noise_size, device=self._config.device)

    @property
    def fixed_labels(self) -> torch.Tensor:
        return torch.randint(0, self._config.num_classes, (64,), device=self._config.device)

    def __call__(self, optimizer: Optimizer) -> None:
        ...

    def train_discriminator(self, optimizer: Optimizer, img_info: ImageInfoDTO) -> TrainDiscriminatorOutDTO:
        ...

    def train_generator(self, optimizer: Optimizer,
                        train_discriminator_out: TrainDiscriminatorOutDTO) -> TrainGeneratorOutDTO:
        ...

    @classmethod
    def plot_losses(cls, all_errors: LossErrors, title: str = 'WGAN') -> None:
        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(y=all_errors.err_discriminator_x, mode='lines', name='D_real'), row=1, col=1)
        fig.add_trace(go.Scatter(y=all_errors.err_discriminator_z, mode='lines', name='D_fake'), row=1, col=1)
        fig.add_trace(go.Scatter(y=all_errors.err_generator, mode='lines', name='G'), row=1, col=1)

        if all_errors.gp_history:
            fig.add_trace(go.Scatter(y=all_errors.gp_history, mode='lines', name='Gradient Penalty'), row=1, col=1)

        fig.update_layout(title=title, xaxis_title='Итерации', yaxis_title='Потери', width=800, height=400)
        fig.show("png")

    @classmethod
    def _log_errors(
            cls,
            errors: LossErrors,
            train_discriminator_out: TrainDiscriminatorOutDTO,
            train_generator_out: TrainGeneratorOutDTO,
    ) -> None:
        errors.err_discriminator_x.append(train_discriminator_out.err_discriminator_real.item())
        errors.err_discriminator_z.append(train_discriminator_out.err_discriminator_fake.item())
        errors.err_generator.append(train_generator_out.err_generator.item())
        if train_discriminator_out.gradient_penalty is not None:
            errors.gp_history.append(train_discriminator_out.gradient_penalty.item())

    def _log_to_tensorboard(
            self,
            global_step: int,
            train_discriminator_out: TrainDiscriminatorOutDTO,
            train_generator_out: TrainGeneratorOutDTO,
    ) -> None:
        self._writer.add_scalar('Loss/Discriminator_real', train_discriminator_out.err_discriminator_real.item(),
                                global_step)
        self._writer.add_scalar('Loss/Discriminator_fake', train_discriminator_out.err_discriminator_fake.item(),
                                global_step)
        self._writer.add_scalar('Loss/Generator', train_generator_out.err_generator.item(), global_step)

    def _log_generated_images(self, global_step: int) -> None:
        if global_step % self._config.log_image_every == 0:
            with torch.no_grad():
                fake_images = self._generator(self.fixed_noise, self.fixed_labels).detach().cpu()

            grid = vutils.make_grid(fake_images.view(-1, *self._config.image_shape), normalize=True)
            self._writer.add_image('Generated Images', grid, global_step)

    def _get_images_info(self, images: torch.Tensor, labels: torch.Tensor) -> ImageInfoDTO:
        return ImageInfoDTO(
            batch_size=images.size(0),
            real_images=images.view(images.size(0), -1).to(self._config.device),
            labels=labels.to(self._config.device),
        )

    def _log_mean_losses(
        self,
        epoch: int,
        epoch_errors: LossErrors,
    ) -> None:
        # Логирование средних потерь за эпоху
        avg_err_discriminator_real = sum(epoch_errors.err_discriminator_x) / len(epoch_errors.err_discriminator_x)
        avg_err_discriminator_fake = sum(epoch_errors.err_discriminator_z) / len(epoch_errors.err_discriminator_z)
        avg_err_generator = sum(epoch_errors.err_generator) / len(epoch_errors.err_generator)

        avg_gp = None
        if epoch_errors.gp_history:
            avg_gp = sum(epoch_errors.gp_history) / len(epoch_errors.gp_history)

        self._writer.add_scalar('Epoch/Loss_D_real', avg_err_discriminator_real, epoch)
        self._writer.add_scalar('Epoch/Loss_D_fake', avg_err_discriminator_fake, epoch)
        self._writer.add_scalar('Epoch/Loss_G', avg_err_generator, epoch)

        if avg_gp is not None:
            self._writer.add_scalar('Epoch/Loss_D_GP', avg_gp, epoch)

        if avg_gp is not None:

            logger.info(f'Epoch [{epoch + 1}/{self._config.num_epochs}] '
                        f'Loss D_real: {avg_err_discriminator_real:.4f}, '
                        f'Loss D_fake: {avg_err_discriminator_fake:.4f}, '
                        f'Loss G: {avg_err_generator:.4f}'
                        f'Loss GP: {avg_gp:.4f}'
                        )
        else:
            logger.info(f'Epoch [{epoch + 1}/{self._config.num_epochs}] '
                        f'Loss D_real: {avg_err_discriminator_real:.4f}, '
                        f'Loss D_fake: {avg_err_discriminator_fake:.4f}, '
                        f'Loss G: {avg_err_generator:.4f}'
                        )

    def _log_generated_progress(self, all_errors: LossErrors, epoch_errors: LossErrors, epoch: int) -> None:
        # Визуализация прогресса генератора после каждой эпохи (опционально)
        with torch.no_grad():
            fake_images = self._generator(self.fixed_noise, self.fixed_labels).detach().cpu()
        grid = vutils.make_grid(fake_images.view(-1, *self._config.image_shape), normalize=True)

        self._writer.add_image('Generated Images Epoch', grid, epoch)

        all_errors.err_generator.extend(epoch_errors.err_generator)
        all_errors.err_discriminator_x.extend(epoch_errors.err_discriminator_x)
        all_errors.err_discriminator_z.extend(epoch_errors.err_discriminator_z)

        if epoch_errors.gp_history:
            all_errors.gp_history.extend(epoch_errors.gp_history)


class WGANClipTrainer(WGANTrainerProtocol):
    __slots__ = ('_generator', '_discriminator', '_writer', '_config')

    def __call__(self, optimizer: Optimizer) -> None:
        self._generator.train()
        self._discriminator.train()

        global_step = 0  # Переменная для отслеживания глобального шага

        all_errors = LossErrors()
        for epoch in range(self._config.num_epochs):
            epoch_errors = LossErrors()  # Сохраняем ошибки для визуализации

            images: torch.Tensor
            for iter_id, (images, labels) in enumerate(self._dataloader):
                img_info: ImageInfoDTO = self._get_images_info(images=images, labels=labels)

                #  Обучение Дискриминатора
                train_discriminator_out = self.train_discriminator(optimizer=optimizer, img_info=img_info)

                # Обучение Генератора
                train_generator_out = self.train_generator(optimizer=optimizer,
                                                           train_discriminator_out=train_discriminator_out)

                # Логирование ошибок
                self._log_errors(
                    errors=epoch_errors,
                    train_discriminator_out=train_discriminator_out,
                    train_generator_out=train_generator_out,
                )

                # Логирование в TensorBoard после каждой итерации
                self._log_to_tensorboard(
                    global_step=global_step,
                    train_discriminator_out=train_discriminator_out,
                    train_generator_out=train_generator_out,
                )

                global_step += 1  # Увеличиваем глобальный шаг

                # Логирование сгенерированных изображений каждые N итераций
                self._log_generated_images(global_step=global_step)

            # Логирование средних потерь за эпоху
            self._log_mean_losses(epoch=epoch, epoch_errors=epoch_errors)

            # Визуализация прогресса генератора после каждой эпохи (опционально)
            self._log_generated_progress(all_errors=all_errors, epoch_errors=epoch_errors, epoch=epoch)

        self.plot_losses(all_errors=all_errors, title='WGAN with Weight Clipping')

    def train_discriminator(self, optimizer: Optimizer, img_info: ImageInfoDTO) -> TrainDiscriminatorOutDTO:
        optimizer.discriminator.zero_grad()

        # Реальные изображения
        real_validity = self._discriminator(img_info.real_images, img_info.labels)
        err_discriminator_real = -torch.mean(real_validity)

        # Сгенерированные изображения
        noise = torch.randn(img_info.batch_size, self._config.noise_size, device=self._config.device)
        gen_labels = torch.randint(0, self._config.num_classes, (img_info.batch_size,), device=self._config.device)
        fake_images = self._generator(noise, gen_labels)
        fake_validity = self._discriminator(fake_images.detach(), gen_labels)
        err_discriminator_fake = torch.mean(fake_validity)

        # Итоговая ошибка дискриминатора
        err_discriminator_total = err_discriminator_real + err_discriminator_fake
        err_discriminator_total.backward()
        optimizer.discriminator.step()

        # Клиппинг весов дискриминатора
        self._make_step_discriminator()

        return TrainDiscriminatorOutDTO(
            err_discriminator_real=err_discriminator_real,
            err_discriminator_fake=err_discriminator_fake,
            fake_images=fake_images,
            gen_labels=gen_labels
        )

    def train_generator(self, optimizer: Optimizer,
                        train_discriminator_out: TrainDiscriminatorOutDTO) -> TrainGeneratorOutDTO:
        optimizer.generator.zero_grad()

        # Сгенерированные изображения для генератора
        gen_validity = self._discriminator(
            train_discriminator_out.fake_images,
            train_discriminator_out.gen_labels,
        )
        err_generator = -torch.mean(gen_validity)
        err_generator.backward()
        optimizer.generator.step()

        return TrainGeneratorOutDTO(err_generator=err_generator)

    def _make_step_discriminator(self) -> None:
        for parameter in self._discriminator.parameters():
            parameter.data.clamp_(-self._config.weight_clip, self._config.weight_clip)


class WGANGPTrainer(WGANTrainerProtocol):
    __slots__ = ('_generator', '_discriminator', '_writer', '_config')

    def __call__(self, optimizer: Optimizer) -> None:
        self._generator.train()
        self._discriminator.train()

        global_step = 0  # Переменная для отслеживания глобального шага

        all_errors = LossErrors()
        for epoch in range(self._config.num_epochs):
            epoch_errors = LossErrors()

            images: torch.Tensor
            for iter_id, (images, labels) in enumerate(self._dataloader):
                img_info: ImageInfoDTO = self._get_images_info(images=images, labels=labels)

                #  Обучение Дискриминатора
                train_discriminator_out = self.train_discriminator(optimizer=optimizer, img_info=img_info)

                # Обучение Генератора
                train_generator_out = self.train_generator(optimizer=optimizer,
                                                           train_discriminator_out=train_discriminator_out)
                # Логирование ошибок
                self._log_errors(
                    errors=epoch_errors,
                    train_discriminator_out=train_discriminator_out,
                    train_generator_out=train_generator_out,
                )

                # Логирование в TensorBoard после каждой итерации
                self._log_to_tensorboard(
                    global_step=global_step,
                    train_discriminator_out=train_discriminator_out,
                    train_generator_out=train_generator_out,
                )

                global_step += 1  # Увеличиваем глобальный шаг

                # Логирование сгенерированных изображений каждые N итераций
                self._log_generated_images(global_step=global_step)

                # Логирование средних потерь за эпоху
            self._log_mean_losses(epoch=epoch, epoch_errors=epoch_errors)

            # Визуализация прогресса генератора после каждой эпохи (опционально)
            self._log_generated_progress(all_errors=all_errors, epoch_errors=epoch_errors, epoch=epoch)

        self.plot_losses(all_errors=all_errors, title='WGAN Gradient Penalty')

    def train_discriminator(self, optimizer: Optimizer, img_info: ImageInfoDTO) -> TrainDiscriminatorOutDTO:
        optimizer.discriminator.zero_grad()

        # Реальные изображения
        real_validity = self._discriminator(img_info.real_images, img_info.labels)
        err_discriminator_real = -torch.mean(real_validity)

        # Сгенерированные изображения
        noise = torch.randn(img_info.batch_size, self._config.noise_size, device=self._config.device)
        gen_labels = torch.randint(0, self._config.num_classes, (img_info.batch_size,), device=self._config.device)
        fake_images = self._generator(noise, gen_labels)
        fake_validity = self._discriminator(fake_images.detach(), gen_labels)
        err_discriminator_fake = torch.mean(fake_validity)

        # Штраф градиентов
        gradient_penalty = self.compute_gradient_penalty(img_info=img_info, fake_images=fake_images)

        # Итоговая ошибка дискриминатора
        err_discriminator_total = err_discriminator_real + err_discriminator_fake + gradient_penalty
        err_discriminator_total.backward()
        optimizer.discriminator.step()

        return TrainDiscriminatorOutDTO(
            err_discriminator_real=err_discriminator_real,
            err_discriminator_fake=err_discriminator_fake,
            gradient_penalty=gradient_penalty,
            fake_images=fake_images,
            gen_labels=gen_labels
        )

    def train_generator(self, optimizer: Optimizer,
                        train_discriminator_out: TrainDiscriminatorOutDTO) -> TrainGeneratorOutDTO:
        optimizer.generator.zero_grad()

        # Сгенерированные изображения для генератора
        gen_validity = self._discriminator(
            train_discriminator_out.fake_images,
            train_discriminator_out.gen_labels,
        )
        err_generator = -torch.mean(gen_validity)
        err_generator.backward()
        optimizer.generator.step()

        return TrainGeneratorOutDTO(err_generator=err_generator)

    def compute_gradient_penalty(
        self,
        img_info: ImageInfoDTO,
        fake_images: torch.Tensor,
    ) -> torch.Tensor:
        # Случайный коэффициент интерполяции между реальными и фейковыми изображениями
        alpha = torch.rand(img_info.batch_size, 1, device=self._config.device)
        alpha = alpha.expand(img_info.real_images.size())

        # Интерполированные изображения
        interpolates = alpha * img_info.real_images + (1 - alpha) * fake_images.detach()
        interpolates.requires_grad_(True)

        # Вычисление выходов дискриминатора на интерполированных изображениях
        d_interpolates = self._discriminator(interpolates, img_info.labels)

        # Градиенты выхода дискриминатора относительно интерполированных изображений
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Вычисление нормы градиентов
        gradients = gradients.view(img_info.batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return self._config.lambda_gp * ((gradient_norm - 1) ** 2).mean()


class DataloaderCreator:
    def __init__(self, data_root: str = "./data", batch_size: int = 64, num_workers: int = 4):
        self._data_root = data_root
        self._batch_size = batch_size
        self._num_workers = num_workers

    def __call__(self) -> DataLoader:
        return self.get_train_fashion_mnist_dataset()

    @classmethod
    def transform(cls) -> Compose:
        return Compose([
            ToTensor(),
            Normalize([0.5], [0.5])
        ])

    def get_train_fashion_mnist_dataset(self) -> DataLoader:
        return DataLoader(
            torchvision.datasets.FashionMNIST(
                root=self._data_root,
                train=True,
                download=True,
                transform=self.transform()
            ),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )
