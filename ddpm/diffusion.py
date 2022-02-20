import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms
from models.unet_ddpm import UNet


class Diffusion():

	def load_model(self):
		url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"

		model = UNet(ch=128, out_ch=3, ch_mult=[1, 1, 2, 2, 4, 4], attn=[5,],
			num_res_blocks=2,dropout=0.0)