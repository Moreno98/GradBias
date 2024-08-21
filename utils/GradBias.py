import torch
import os
from utils.generative_models import Stable_Diffusion, Stable_Diffusion_XL
import utils.VQA as VQA
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from utils.utils import STOP_WORDS
import numpy as np
import datetime
from torch.distributed import init_process_group, destroy_process_group
import utils.utils as utils
import torchvision.transforms as T

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

class GradBias:
    def __init__(
        self, 
        gen_config,
        vqa_info,
        devices, 
        save_info = False,
        loss_interval = 1
    ):
        self.control_tokens = False # consider special tokens (e.g., <EOS>)
        self.sd_devices, self.vqa_devices, self.ranks = utils.setup_gpu(devices, CLIP = not 'llava' in vqa_info['path'])
        self.inference_steps = gen_config['inference_steps']

        assert self.inference_steps % loss_interval == 0, 'INFERENCE_STEPS must be divisible by loss_interval'

        self.save_info = save_info
        self.loss_interval = loss_interval
        
        # Loss INITIALIZATION
        self.loss = self.compute_matching_loss
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.stop_words = STOP_WORDS

        # Generator INITIALIZATION
        self.sd = eval(gen_config['class'])(
            gen_config = gen_config,
            device=self.sd_devices,
            n_images=1,
            inference_steps=self.inference_steps,
            safe_checker=False
        )
        
        self.sd.dm.detach_output = False
        self.sd_tokenizer = self.sd.dm.tokenizer
        self.clear_memory = False
        if 'xl' in gen_config['version']:
            if len(self.vqa_devices) == 3:
                self.sd.dm.vae = self.sd.dm.vae.to(self.vqa_devices[-1])
                self.sd.dm.text_encoder = self.sd.dm.text_encoder.to(self.vqa_devices[-3])
                self.sd.dm.text_encoder_2 = self.sd.dm.text_encoder_2.to(self.vqa_devices[-2])
            else:
                self.sd.dm.vae = self.sd.dm.vae.to(self.vqa_devices[-1])
                self.sd.dm.text_encoder = self.sd.dm.text_encoder.to(self.vqa_devices[-1])
                self.sd.dm.text_encoder_2 = self.sd.dm.text_encoder_2.to(self.vqa_devices[-1])
            self.clear_memory = True
            torch.cuda.empty_cache()
        # VQA INITIALIZATION
        if 'llava' in vqa_info['path']:
            max_memory = utils.get_max_memory(self.vqa_devices)
            self.vqa_name = 'llava'
            self.vqa_wrapper = VQA.Llava(self.vqa_devices, vqa_info['path'], requires_grad=True, max_memory=max_memory)
            self.vqa_model = self.vqa_wrapper.get_model()
            self.vqa_tokenizer = self.vqa_wrapper.tokenizer
            # self.vqa_model = self.vqa_model.to(dtype = torch.float32)

            self.outputs = []
            self.hooks = []
            self.hooks.append(
                self.vqa_model.lm_head.register_forward_hook(self.save_output)
            )
        else:
            self.vqa_name = 'clip'
            self.vqa_wrapper = VQA.Clip_model(self.vqa_devices[0], vqa_info['path'], requires_grad=True)
            self.vqa_model = self.vqa_wrapper.get_model()
            self.clip_transform = T.Compose(
                [
                    T.Resize(336, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
                    T.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
                ]
            )
    
    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_path(self, path):
        self.path = path
    
    def post_process_images_llava(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        # resize to 336x504
        image = torch.nn.functional.interpolate(image, (336, 336), mode='bilinear', align_corners=False)
        # center crop {'height': 336, 'width': 336}
        image = image.squeeze(0).cpu().permute(1, 2, 0)
        image = (image - torch.tensor(OPENAI_CLIP_MEAN, dtype=image.dtype)) / torch.tensor(OPENAI_CLIP_STD, dtype=image.dtype)

        return image

    def post_process_images_clip(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0).cpu()
        image = self.clip_transform(image)
        return image

    def save_image(self, image, filename):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0).cpu().detach().permute(1, 2, 0).numpy().astype("float32")
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image)
        image.save(filename)

    def token_mapping(self, prompt, text_input_tokens, control_tokens = False):
        words = prompt.lower().split()
        i = 0
        current_decoding = ''
        current_tokens = []
        mapping = []
        for token in text_input_tokens[1:-1]:
            decoded_token = self.sd_tokenizer.decode(token)
            current_decoding += decoded_token
            current_tokens.append(token)
            if current_decoding == words[i]:
                mapping.append(
                    (
                        current_decoding,
                        current_tokens
                    )
                )
                current_decoding = ''
                current_tokens = []
                i += 1

        if control_tokens:
            # Add start token to the start of word_token_mapping
            mapping.insert(0, (self.sd_tokenizer.decode(text_input_tokens[0]), [text_input_tokens[0]]))
            # Add end token to the end of word_token_mapping
            mapping.append((self.sd_tokenizer.decode(text_input_tokens[-1]), [text_input_tokens[-1]]))
        return mapping

    def check_occurances(self, word_level_gradients):
        word_occurances = {}
        for word in word_level_gradients:
            if word[0] not in word_occurances:
                word_occurances[word[0]] = 0
            word_occurances[word[0]] += 1
            if word_occurances[word[0]] > 1:
                word_level_gradients[word_level_gradients.index(word)] = (f'{word[0]}_{word_occurances[word[0]]}', word[1])
        return word_level_gradients

    '''
    Run the pipeline to compute the gradients of the words in the prompt
    Args:
        prompt: str, the user prompt to be used
        question: str, the question to be used (for Llava)
        choices: list, the choices to be used
    Returns:
        word_gradients_mean: dict, the mean of the gradients for each word in the prompt
    '''
    def run_pipeline(self, prompt, question, choices): 
        generator = self.sd.generate_images_grad(
            [prompt], 
            loss_interval=self.loss_interval
        )

        word_gradients_mean = {}
        i = 1
        for images, prompt_embeds, text_input_ids, attention_mask_input in generator:
            prompt_embeds.retain_grad()
            # set the detach_output to False as we need to compute the gradients
            # we will set it to True every time we compute the loss -- every loss_interval steps
            self.sd.dm.detach_output = False

            # compute the gradients every loss_interval steps
            # otherwise continue with the denoising process
            if i%self.loss_interval == 0:
                if i == self.inference_steps and self.save_info:
                    self.save_image(images, os.path.join(self.path, 'image.png'))
    
                if 'llava' in self.vqa_model.__class__.__name__.lower():
                    image = self.post_process_images_llava(images)
                    images = image.permute(2,0,1).unsqueeze(0).to(self.vqa_devices[0])
                else:
                    # cast images to pil image
                    image = self.post_process_images_clip(images)
                    images = image.unsqueeze(0).to(self.vqa_devices[0])

                image_gradients = []
                # we may consider multiple templates to query the vqa model
                for choice_template in choices:
                    image_detached = images.detach().clone()
                    image_detached.requires_grad = True
                    choices = [choice.lower() for choice in choice_template]

                    # query the VLM model with the image and the bias related information
                    # Please note: when considering CLIP the question is not used
                    vqa_output = self.vqa_wrapper.vqa(
                        image=image_detached, 
                        question=question,
                        choices=choices
                    )

                    # compute the loss and the gradients for each image
                    self.compute_loss(vqa_output)
                    # save the gradients
                    image_gradients.append(image_detached.grad)
                
                # compute the average of the gradients
                image_grads_avg = torch.stack(image_gradients)
                assert len(image_grads_avg.shape) == 5, 'Number of choices and gradients must be the same'
                image_grads_avg = image_grads_avg.mean(dim=0)
                # backward pass -- backward through the generator and compute the gradients of the input tokens
                self.backward(images, image_grads_avg)

                indices = (attention_mask_input[0] == 1).nonzero().cpu().squeeze()
                input_grads = prompt_embeds.grad[1][indices].mean(dim=-1)
                input_tokens = text_input_ids[0][indices]

                # get the mapping of words - tokens
                word_token_mapping = self.token_mapping(prompt, input_tokens, control_tokens=self.control_tokens)
                # Get the absolute values of gradients
                abs_grads = torch.abs(input_grads)

                if self.control_tokens:
                    tokens_idx = 0
                else:
                    tokens_idx = 1

                word_level_gradients_mean = []

                for mapping in word_token_mapping:
                    if mapping[0] not in self.stop_words:
                        word_gradient_mean = abs_grads[tokens_idx:tokens_idx+len(mapping[1])].mean().item()
                        word_level_gradients_mean.append((mapping[0], word_gradient_mean))
                    tokens_idx += len(mapping[1])

                # identify the words occuring multiple times
                # replace them with a text identifier (e.g., {word}_{occurance})
                # e.g., 'elephant', 'elephant_2'
                word_level_gradients_mean = self.check_occurances(word_level_gradients_mean)  

                # Sort the word_level_gradients              
                sorted_grads_mean = sorted(word_level_gradients_mean, key=lambda x: x[1], reverse=True)

                for word in sorted_grads_mean:
                    if word[0] not in word_gradients_mean:
                        word_gradients_mean[word[0]] = [word[1]]
                    else:
                        word_gradients_mean[word[0]].append(word[1])

                self.sd.dm.detach_output = True
                # torch.cuda.empty_cache()
            i += 1

        if self.save_info:
            self.save_word_level_bias(word_gradients_mean, prompt, self.path, 'mean')
        
        # clean memory  
        torch.cuda.empty_cache()
        return word_gradients_mean

    # Save the word level gradients to a file
    def save_word_level_bias(self, word_gradients, prompt, path, reduction):
        with open(os.path.join(path, f'word_level_bias_{reduction}.txt'), 'w') as f:
            f.write(prompt + '\n')
            for word, grads in word_gradients.items():
                f.write(f'{word}: {grads}\n')

    def save_output(self, module, input, output):
        self.outputs.append(output[:, -1, :])
    
    def clear_outputs(self):
        self.outputs = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.clear()
            hook.remove()

    def compute_matching_loss(self, outputs):
        y = torch.argmax(outputs, dim=-1)
        return self.criterion(outputs, y)

    def criterion(self, output, y):
        return torch.nn.CrossEntropyLoss()(output, y)
    
    def backward(self, image, gradient):
        self.sd.dm.unet.zero_grad()
        self.sd.dm.vae.zero_grad()
        self.sd.dm.text_encoder.zero_grad()
        if self.clear_memory:
            self.sd.dm.text_encoder_2.zero_grad()
        image.backward(gradient=gradient)

    def compute_loss(self, vqa_output):
        if self.vqa_name == 'llava':
            outputs = torch.cat(self.outputs, dim=0)
        else:
            outputs = vqa_output
        
        loss = self.loss(outputs)
        self.sd.dm.unet.zero_grad()
        self.sd.dm.vae.zero_grad()
        self.sd.dm.text_encoder.zero_grad()
        if self.clear_memory:
            self.sd.dm.text_encoder_2.zero_grad()
        self.vqa_model.zero_grad()
        loss.backward()

        if self.vqa_name == 'llava':
            self.clear_outputs()        
    