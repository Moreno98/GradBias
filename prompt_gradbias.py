from utils.GradBias import GradBias
from utils.config import GENERATORS, VQA_MODELS
import torch
from utils.pipeline_stable_diffusion import SDPipeline, SDXLPipeline
import numpy as np
import argparse
from argparse import RawTextHelpFormatter

def main(opt, caption, question, choices):
    gpus = torch.cuda.device_count()
    devices = [f'cuda:{i}' for i in range(gpus)]
    grad_bias = GradBias(
        gen_config=opt['generator'],
        vqa_info=opt['vqa'],
        devices=devices,
        save_info=opt['save_info'],
        loss_interval=opt['loss_interval']
    )
    
    word_gradients_mean = grad_bias.run_pipeline(
        prompt=caption,
        question=question,
        choices=choices
    )

    words = {}
    for word, gradients in word_gradients_mean.items():
        words[word] = np.array(gradients).mean().item()    

    # rank words by their gradients
    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    for word, gradient in sorted_words:
        print(f'{word}: {gradient}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GradBias', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--vqa_model', choices=list(VQA_MODELS.keys()), default='llava-1.5-13b', help="vqa model to use")  
    opt = vars(parser.parse_args())
    opt = {
        'generator': GENERATORS['sd-xl'],
        'seeds': [0],
        'use_gradcam': False,
        'vqa': {'path': VQA_MODELS[opt['vqa_model']][1]},
        'loss': 'matching_loss',
        'save_info': False,
        'loss_interval': 1,
        'accumulate_grad': False,
        'vqa_model': opt['vqa_model']
    }
    opt['generator']['model_class'] = SDXLPipeline

    # set the caption here
    caption = ''

    # set the choices here
    # for CLIP, the choices should be in the form of 'a photo of a <choice>'
    # for llava model, the choices should be in the form of '<choice>'
    choices = [
        [
            '', 
            '',
        ]  
    ] 

    # set the question here
    # the question is required only by CLIP. Leave empty for llava.
    # The question should be in the form of '<question> Answer with one word.'
    # The question should represent the bias you want to evaluate. 
    # For example, if you want to evaluate gender, the question can be "What is the gender of the person? Answer with one word."
    question = ""

    main(opt, caption, question, choices)