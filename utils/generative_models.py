from torch import autocast
import sys
import torch
from diffusers import EulerDiscreteScheduler
from diffusers.utils import torch_utils

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def dummy_checker(images, **kwargs):
    return images, [False]*len(images)

# diffusion xl model class
class Stable_Diffusion_XL():
    def __init__(
        self,
        gen_config,
        device,
        n_images,
        inference_steps,
        safe_checker = False
    ):
        # base diffusion xl model
        self.dm = gen_config['model_class'].from_pretrained(
            gen_config['version'], 
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)

        # self.dm.enable_model_cpu_offload()
        self.dm.set_progress_bar_config(disable=True)
        if not safe_checker:
            self.dm.safety_checker = dummy_checker
        
        self.base_model_only = gen_config['base_model_only']
        if not self.base_model_only:
            # refiner
            self.refiner = gen_config['model_class'].from_pretrained(
                gen_config['refiner'],
                text_encoder_2=self.dm.text_encoder_2,
                vae=self.dm.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(device)
            # self.refiner.enable_model_cpu_offload()
            self.refiner.set_progress_bar_config(disable=True)
            if not safe_checker:
                self.refiner.safety_checker = dummy_checker
        
        self.high_noise_frac = 0.8
        self.inference_steps = inference_steps
        self.n_images = n_images
        self.neg_prompt = ['cartoon, painting, black and white, duplicate, extra legs, longbody, low resolution, bad anatomy, missing fingers, extra digit, fewer digits, cropped, low quality']*1
        # self.neg_prompt = ['']*1
        # speed up stable diffusion xl
        # self.dm = deepspeed.initialize(self.dm)
        # self.refiner = deepspeed.initialize(self.refiner)
    
    @torch.no_grad()
    def generate_images(self, prompt, seeds):
        if not self.base_model_only:
            images = self.dm(
                prompt, 
                negative_prompt=self.neg_prompt,
                num_inference_steps = self.inference_steps,
                num_images_per_prompt=self.n_images,
                denoising_end=self.high_noise_frac,
                output_type="latent",
            ).images
            images = self.refiner(
                prompt=prompt,
                negative_prompt=self.neg_prompt,
                num_inference_steps=self.inference_steps,
                num_images_per_prompt=self.n_images,
                denoising_start=self.high_noise_frac,
                image=images, 
            ).images
        else:
            return self.dm(
                prompt, 
                negative_prompt=self.neg_prompt,
                num_inference_steps = self.inference_steps,
                num_images_per_prompt=self.n_images,
                latents=self.prepare_latents(seeds)
            ).images
    
    def prepare_latents(self, seeds):
        latents = []
        num_channels_latents = self.dm.unet.config.in_channels
        height = self.dm.unet.config.sample_size
        width = self.dm.unet.config.sample_size
        shape = (1, num_channels_latents, height, width)
        for seed in seeds:
            torch.manual_seed(seed)
            latents.append(
                torch_utils.randn_tensor(
                    shape=shape,
                    device=self.dm.device,
                    dtype=torch.float16,
                )
            )
        latents = torch.cat(latents, dim=0)   
        return latents
    
    def generate_images_grad(
        self, 
        prompt, 
        detach_output_at_each_step=False,
        loss_interval=1
    ):
        return self.dm(
            prompt, 
            negative_prompt=self.neg_prompt,
            num_inference_steps = self.inference_steps,
            num_images_per_prompt=self.n_images,
            detach_output_at_each_step=detach_output_at_each_step,
            loss_interval=loss_interval
        )

# diffusion model class (for 1.5 and 2.0)
class Stable_Diffusion():
    def __init__(
        self,
        gen_config,
        device,
        n_images,
        inference_steps,
        safe_checker = False
    ):
        scheduler = EulerDiscreteScheduler.from_pretrained(gen_config['version'], subfolder="scheduler")
        # diffusion model
        self.dm = gen_config['model_class'].from_pretrained(
            gen_config['version'], 
            scheduler=scheduler,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        # self.dm.enable_model_cpu_offload()
        self.dm.set_progress_bar_config(disable=True)
        if not safe_checker:
            self.dm.safety_checker = dummy_checker
        
        self.inference_steps = inference_steps
        self.n_images = n_images
        self.neg_prompt = ['cartoon, painting, black and white, duplicate, extra legs, longbody, low resolution, bad anatomy, missing fingers, extra digit, fewer digits, cropped, low quality']*1

    def generate_images_grad(self, prompt, detach_output_at_each_step=False, loss_interval=1):
        return self.dm(
            prompt, 
            negative_prompt=self.neg_prompt,
            num_inference_steps=self.inference_steps,
            num_images_per_prompt=self.n_images,
            detach_output_at_each_step=detach_output_at_each_step,
            loss_interval=loss_interval
        )

    def prepare_latents(self, seeds):
        latents = []
        num_channels_latents = self.dm.unet.config.in_channels
        height = self.dm.unet.config.sample_size
        width = self.dm.unet.config.sample_size
        shape = (1, num_channels_latents, height, width)
        for seed in seeds:
            torch.manual_seed(seed)
            latents.append(
                torch_utils.randn_tensor(
                    shape=shape,
                    device=self.dm.device,
                    dtype=torch.float16,
                )
            )
        latents = torch.cat(latents, dim=0)   
        return latents

    @torch.no_grad()
    def generate_images(self, prompt, seeds):
        return self.dm(
            prompt, 
            negative_prompt=self.neg_prompt,
            num_inference_steps=self.inference_steps,
            num_images_per_prompt=self.n_images,
            latents=self.prepare_latents(seeds)
        ).images
