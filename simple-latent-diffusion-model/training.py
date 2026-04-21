import torch
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os

from clip.models.clip import CLIP
from diffusion_model.models.clip_latent_diffusion_model import CLIPLatentDiffusionModel

from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer
from helper.loader import Loader
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.unet import Unet
from diffusion_model.sampler.ddim import DDIM
from diffusion_model.network.unet_wrapper import UnetWrapper
from helper.cond_encoder import ClassEncoder

IMAGE_SHAPE = (3, 32, 32)
CONFIG_PATH = './configs/cifar10_config.yaml'
VAE_FILE_NAME = './auto_encoder/check_points/vae'
DM_FILE_NAME = './diffusion_model/check_points/ldm'

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t' + (
        f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU'))

    data_generator = DataGenerator()
    data_loader = data_generator.cifar10(batch_size=128)
    # data_loader = data_generator.composite('./datasets/image/', './datasets/json/')
    painter = Painter()
    loader = Loader()

    vae = VariationalAutoEncoder(CONFIG_PATH)
    checkpoint = torch.load("./auto_encoder/check_points/vae.pth")

    vae.load_state_dict(checkpoint['model_state_dict'])
    # trainer = Trainer(vae, loss_fn = vae.loss)
    # trainer.train(data_loader, 1000, VAE_FILE_NAME, True)

    sampler = DDIM(CONFIG_PATH)
    cond_encoder=ClassEncoder(CONFIG_PATH)
    unet=UnetWrapper(Unet, CONFIG_PATH,cond_encoder=cond_encoder)
    dm = LatentDiffusionModel(unet, sampler, vae)
    trainer = Trainer(dm, dm.loss)
    trainer.train(data_loader, 1000, DM_FILE_NAME, False)
    # clip = CLIP('./configs/composite_clip_config.yaml')
    #
    # sampler = DDIM(CONFIG_PATH)
    # network = ConditionalUnetworkWrapper(CONFIG_PATH)
    # dm = CLIPLatentDiffusionModel(network, sampler, vae, clip, IMAGE_SHAPE)
    # trainer = Trainer(dm, dm.loss)
    # trainer.train(data_loader, 100, DM_FILE_NAME, False)
