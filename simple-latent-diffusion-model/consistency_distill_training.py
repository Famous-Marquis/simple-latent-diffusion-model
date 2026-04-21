import argparse
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from diffusion_model.network.unet import Unet
from diffusion_model.network.unet_wrapper import UnetWrapper
from helper.cond_encoder import ClassEncoder
from helper.data_generator import DataGenerator
from helper.ema import EMA


CONFIG_PATH = './configs/cifar10_config.yaml'
VAE_CKPT = './auto_encoder/check_points/vae.pth'
TEACHER_CKPT = './diffusion_model/check_points/ldm.pth'
DISTILL_CKPT = './diffusion_model/check_points/ldm_consistency_1step.pth'


class ConsistencyDistiller:
    """Consistency distillation for class-conditional CIFAR10 latent diffusion.

    Teacher predicts epsilon and provides a deterministic DDIM-like transition x_t -> x_s.
    Student learns a consistency mapping f_theta(x_t, t, y) ~= f_theta(x_s, s, y),
    where f_theta returns x0 estimate from epsilon prediction.
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        alpha_bar: torch.Tensor,
        num_scales: int,
        teacher_loss_weight: float,
        device: torch.device,
    ):
        self.teacher = teacher.eval().to(device)
        self.student = student.to(device)
        self.alpha_bar = alpha_bar.to(device)
        self.teacher_loss_weight = teacher_loss_weight
        self.device = device

        for p in self.teacher.parameters():
            p.requires_grad = False

        # t_0 > t_1 > ... > t_{N-1} = 0
        self.timesteps = torch.linspace(
            self.alpha_bar.size(0) - 1,
            0,
            steps=num_scales,
            dtype=torch.long,
            device=device,
        )

    def _predict_x0(self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        eps_hat = model(x=x_t, t=t, y=y)
        alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_t) * eps_hat) / torch.sqrt(alpha_t)
        return x0_hat, eps_hat

    @torch.no_grad()
    def _teacher_step(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, y: torch.Tensor):
        x0_hat, eps_hat = self._predict_x0(self.teacher, x_t, t, y)
        alpha_s = self.alpha_bar[s].view(-1, 1, 1, 1)
        # Deterministic DDIM transition (eta = 0)
        x_s = torch.sqrt(alpha_s) * x0_hat + torch.sqrt(1.0 - alpha_s) * eps_hat
        return x_s, x0_hat

    def loss(self, latent_x0: torch.Tensor, y: torch.Tensor):
        bsz = latent_x0.size(0)

        # Pick neighbor pair from discretized schedule: t > s
        i = torch.randint(0, self.timesteps.size(0) - 1, (bsz,), device=self.device)
        t = self.timesteps[i]
        s = self.timesteps[i + 1]

        eps = torch.randn_like(latent_x0)
        alpha_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_t) * latent_x0 + torch.sqrt(1.0 - alpha_t) * eps

        with torch.no_grad():
            x_s, teacher_x0_t = self._teacher_step(x_t, t, s, y)

        student_x0_t, _ = self._predict_x0(self.student, x_t, t, y)
        student_x0_s, _ = self._predict_x0(self.student, x_s, s, y)

        consistency_loss = F.mse_loss(student_x0_t, student_x0_s)
        teacher_reg_loss = F.mse_loss(student_x0_t, teacher_x0_t)
        total_loss = consistency_loss + self.teacher_loss_weight * teacher_reg_loss

        metrics = {
            'loss': total_loss.item(),
            'consistency': consistency_loss.item(),
            'teacher_reg': teacher_reg_loss.item(),
        }
        return total_loss, metrics


@torch.no_grad()
def sample_one_step(student: torch.nn.Module, vae: VariationalAutoEncoder, alpha_bar: torch.Tensor, y: torch.Tensor):
    """One-step generation from Gaussian latent x_T -> x0_hat."""
    device = y.device
    n_samples = y.size(0)
    t = torch.full((n_samples,), alpha_bar.size(0) - 1, dtype=torch.long, device=device)

    z_t = torch.randn(n_samples, vae.embed_dim, *vae.decoder.z_shape[2:], device=device)
    eps_hat = student(x=z_t, t=t, y=y)

    alpha_t = alpha_bar[t].view(-1, 1, 1, 1)
    z0_hat = (z_t - torch.sqrt(1.0 - alpha_t) * eps_hat) / torch.sqrt(alpha_t)
    image = vae.decode(z0_hat)
    return image


def build_models(config_path: str, device: torch.device):
    vae = VariationalAutoEncoder(config_path).to(device)
    vae_ckpt = torch.load(VAE_CKPT, map_location=device, weights_only=True)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    teacher_cond = ClassEncoder(config_path)
    teacher = UnetWrapper(Unet, config_path, cond_encoder=teacher_cond).to(device)

    student_cond = ClassEncoder(config_path)
    student = UnetWrapper(Unet, config_path, cond_encoder=student_cond).to(device)

    teacher_ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=True)
    teacher.load_state_dict(teacher_ckpt['model_state_dict'])

    # Initialize student from teacher to stabilize distillation.
    student.load_state_dict(deepcopy(teacher.state_dict()))

    return vae, teacher, student


def train(args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}' + (f'\t{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ''))

    vae, teacher, student = build_models(args.config, device)

    # reuse teacher sampler alpha_bar via config-equivalent construction
    from diffusion_model.sampler.ddim import DDIM
    sampler = DDIM(args.config).to(device)

    distiller = ConsistencyDistiller(
        teacher=teacher,
        student=student,
        alpha_bar=sampler.alpha_bar,
        num_scales=args.num_scales,
        teacher_loss_weight=args.teacher_loss_weight,
        device=device,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(student, beta=args.ema_beta).to(device)

    data_loader: DataLoader = DataGenerator().cifar10(batch_size=args.batch_size)

    global_step = 0
    best_loss = float('inf')

    student.train()
    for epoch in range(1, args.epochs + 1):
        running = {'loss': 0.0, 'consistency': 0.0, 'teacher_reg': 0.0}
        progress = tqdm(data_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)

        for step, (images, labels) in enumerate(progress, start=1):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                latent_x0 = vae.encode(images).sample()

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = distiller.loss(latent_x0=latent_x0, y=labels)
            loss.backward()
            # no gradient clipping by request
            optimizer.step()
            ema.update()

            global_step += 1
            for k in running:
                running[k] += metrics[k]

            if global_step % args.log_interval == 0:
                step_avg = {k: running[k] / step for k in running}
                progress.set_postfix(
                    loss=f"{step_avg['loss']:.4f}",
                    consistency=f"{step_avg['consistency']:.4f}",
                    teacher_reg=f"{step_avg['teacher_reg']:.4f}",
                )

        epoch_loss = running['loss'] / len(data_loader)
        print(f"Epoch {epoch} | loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(
                {
                    'model_state_dict': student.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_loss': best_loss,
                    'args': vars(args),
                },
                args.output,
            )
            print(f'Best model saved to {args.output}')

        if epoch % args.sample_every == 0:
            ema.ema_model.eval()
            y = torch.arange(0, min(10, args.sample_batch), device=device)
            with torch.no_grad():
                sample = sample_one_step(ema.ema_model, vae, sampler.alpha_bar.to(device), y=y)
            print(f'1-step sample generated: shape={tuple(sample.shape)}')
            ema.ema_model.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Consistency distillation for CIFAR10 conditional LDM (1-step).')
    parser.add_argument('--config', type=str, default=CONFIG_PATH)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-scales', type=int, default=32, help='Number of discrete consistency timesteps.')
    parser.add_argument('--teacher-loss-weight', type=float, default=1.0)
    parser.add_argument('--ema-beta', type=float, default=0.9999)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--sample-every', type=int, default=10)
    parser.add_argument('--sample-batch', type=int, default=10)
    parser.add_argument('--output', type=str, default=DISTILL_CKPT)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
