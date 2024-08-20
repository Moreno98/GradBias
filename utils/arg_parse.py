from argparse import RawTextHelpFormatter
import argparse
from utils.config import GENERATORS, DATASET_CONFIG, VQA_MODELS, VQA_SETTING, BIASES_TO_CHECK, LLM_CONFIG
import torch
import os
import utils.utils as utils
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils.pipeline_stable_diffusion import SDPipeline, SDXLPipeline
from utils.utils import setup_gpu

EXP_ROOT_PATH = 'methods'
    
# gradient method arg parser
def argparse_gradBias():
    parser = argparse.ArgumentParser(description='GradBias', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--loss', type=str, default='matching_loss', choices=['matching_loss', 'adversarial_loss'], help='Which loss to use')
    parser.add_argument('--generator', type=str, default='sd-2', choices=list(GENERATORS.keys()), help='Generator to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='Seeds to use')
    parser.add_argument('--dataset', type=str, default='coco', choices=list(DATASET_CONFIG.keys()), help='Dataset-related proposed biases to use')
    parser.add_argument('--save_info', action='store_true', help='Save generated image')
    parser.add_argument('--loss_interval', type=int, default=1, help='Interval to compute loss during debiasing process')
    parser.add_argument('--vqa_model', choices=list(VQA_MODELS.keys()), default='llava-1.5-13b', help="vqa model to use")  
    parser.add_argument('--multi_prompts', action='store_true', help='Use multiple prompts for querying the vqa model')
    opt = vars(parser.parse_args())

    opt['vqa'] = {
        'path': VQA_MODELS[opt['vqa_model']][1],
    }
    gpus = torch.cuda.device_count()
    opt['generator_name'] = opt['generator']
    opt['generator'] = GENERATORS[opt['generator']]
    if 'diffusion' in opt['generator']['class'].lower():
        if 'xl' in opt['generator']['class'].lower():
            opt['generator']['model_class'] = SDXLPipeline
            if 'llava' in opt['vqa_model']:
                opt['N_GPU_PER_PROC'] = 4
            else:
                opt['N_GPU_PER_PROC'] = 2
        elif '2' in opt['generator']['class'].lower():
            opt['generator']['model_class'] = SDPipeline
            if 'llava' in opt['vqa_model']:
                opt['N_GPU_PER_PROC'] = 3
            else:
                opt['N_GPU_PER_PROC'] = 1
        else:
            opt['generator']['model_class'] = SDPipeline
            if 'llava' in opt['vqa_model']:
                opt['N_GPU_PER_PROC'] = 2
            else:
                opt['N_GPU_PER_PROC'] = 1

    if opt['multi_prompts'] and 'clip' in opt['vqa_model']:
        opt['choices_templates'] = [
            '',
            'a photo of a {} {}',
            'a picture of a {} {}',
            'a {} {}',
        ]
    else:
        opt['choices_templates'] = [
            ''
        ]

    opt['devices'] = [f'cuda:{i}' for i in range(gpus)]
    opt['save_path'] = os.path.join(EXP_ROOT_PATH, 'GradBias', opt['vqa_model'], opt['dataset'], opt['generator_name'], opt['loss'], str(opt['loss_interval']))
    return opt

def argparse_image_gen_gt():
    parser = argparse.ArgumentParser(description='Compute gt', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--generator', type=str, default='sd-2', choices=list(GENERATORS.keys()), help='Generator to use')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4], help='Seeds to use')
    parser.add_argument('--use_gradcam', type=bool, default=False, help='Use gradcam')
    parser.add_argument('--dataset', type=str, default='coco', choices=list(DATASET_CONFIG.keys()), help='Dataset related proposed biases to use')
    parser.add_argument('--edit_mode', type=str, default='generic', choices=['generic', 'remove'], help='Remove or make generic the words')
    opt = vars(parser.parse_args())
    opt['save_path'] = os.path.join('generated_images', opt['dataset'], opt['generator'], 'gt', opt['edit_mode'])
    os.makedirs(opt['save_path'], exist_ok=True)
    opt['generator'] = GENERATORS[opt['generator']]
    if 'diffusion' in opt['generator']['class'].lower():
        if 'xl' in opt['generator']['class'].lower():
            opt['generator']['model_class'] = StableDiffusionXLPipeline
        else:
            opt['generator']['model_class'] = StableDiffusionPipeline

    opt['n-images'] = len(opt['seeds'])

    if opt['edit_mode'] == 'generic':
        opt['edit_word'] = utils.generic_word
    else:
        opt['edit_word'] = utils.remove_word
    return opt

def argparse_VQA_gt():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', choices=['coco'], help="dataset to use")
    parser.add_argument('--edit_mode', type=str, default='generic', choices=['generic', 'remove'], help='Remove or make generic the words')
    parser.add_argument('--vqa_model', choices=list(VQA_MODELS.keys()), help="vqa model to use")
    parser.add_argument('--generator', choices=list(GENERATORS.keys()), help="Generated images to run the VQA on")
    opt = vars(parser.parse_args())

    opt['dataset_setting'] = DATASET_CONFIG[opt['dataset']]
    opt['dataset_setting']['images_path'] = os.path.join(
        'generated_images',
        opt['dataset'],
        opt['generator'],
        'gt',
        opt['edit_mode']
    )
    opt['dataset_setting']['biases_to_check'] = BIASES_TO_CHECK

    opt['UNK_CLASS'] = VQA_SETTING['UNK_CLASS']
    opt['seed'] = VQA_SETTING['seed']
    opt['vqa_model_name'] = opt['vqa_model']
    opt['vqa_model'] = VQA_MODELS[opt['vqa_model']]

    opt['save_path'] = os.path.join(
        EXP_ROOT_PATH,
        'VQA_gt',
        opt['dataset'],
        opt['vqa_model_name'],
        opt['generator'],
        'gt',
        opt['edit_mode'],
    )
    os.makedirs(opt['save_path'], exist_ok=True)
    opt['file_name'] = 'vqa_answers.json'
    
    return opt

def argparse_VQA_baseline():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', default='coco', choices=['coco'], help="dataset to use")
    parser.add_argument('--vqa_model', choices=list(VQA_MODELS.keys()), help="vqa model to use")
    parser.add_argument('--generator', choices=list(GENERATORS.keys()), help="Generated images to run the VQA on")
    parser.add_argument('--custom_sys', action='store_true', help='Use custom system prompt')
    opt = vars(parser.parse_args())

    opt['dataset_setting'] = DATASET_CONFIG[opt['dataset']]
    opt['dataset_setting']['images_path'] = os.path.join(
        'generated_images',
        opt['dataset'],
        opt['generator'],
        'gt',
        'remove'
    )
    opt['dataset_setting']['biases_to_check'] = BIASES_TO_CHECK

    opt['seed'] = VQA_SETTING['seed']
    opt['vqa_model_name'] = opt['vqa_model']
    opt['vqa_model'] = VQA_MODELS[opt['vqa_model']]


    opt['save_path'] = os.path.join(
        EXP_ROOT_PATH,
        'VQA_baseline',
        opt['dataset'],
        opt['vqa_model_name'],
        opt['generator'],
        'remove',
    )
    os.makedirs(opt['save_path'], exist_ok=True)
    opt['file_name'] = 'vqa_answers.json'
    
    return opt

def argparse_LLM():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--LLM', type=str, choices=list(LLM_CONFIG['LLMs'].keys()), help='Path to the model checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--dataset', choices=['coco'], help="dataset to use")
    opt = vars(parser.parse_args())
    opt['save_path'] = os.path.join(
        EXP_ROOT_PATH,
        'LLM_Baseline',
        opt['dataset'],
        opt['LLM'],
    )
    os.makedirs(opt['save_path'], exist_ok=True)
    opt['LLM_config'] = LLM_CONFIG
    opt['LLM'] = LLM_CONFIG['LLMs'][opt['LLM']]
    opt['dataset_setting'] = DATASET_CONFIG[opt['dataset']]
    opt['dataset_setting']['biases_to_check'] = BIASES_TO_CHECK
    opt['N_GPU_PER_PROC'] = opt['LLM']['N_CUDA']

    opt['file_name'] = 'Rankings.json'
    return opt

def argparse_sintax_tree():
    parser = argparse.ArgumentParser(description='Commands description', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--dataset', choices=['coco'], default='coco', help="dataset to use")
    opt = vars(parser.parse_args())
    opt['save_path'] = os.path.join(
        EXP_ROOT_PATH,
        'Syntax_tree_baseline',
        opt['dataset'],
    )
    os.makedirs(opt['save_path'], exist_ok=True)
    opt['dataset_setting'] = DATASET_CONFIG[opt['dataset']]
    opt['dataset_setting']['biases_to_check'] = BIASES_TO_CHECK
    opt['file_name'] = 'Rankings.json'
    return opt