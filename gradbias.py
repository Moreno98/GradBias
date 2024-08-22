from utils.GradBias import GradBias
import numpy as np
import torch
import utils.arg_parse as arg_parse
from utils.config import DATASET_CONFIG, BIASES_TO_CHECK
import utils.datasets as datasets
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from utils.utils import save_word_level_bias
import utils.utils as utils
from torch.distributed import init_process_group, destroy_process_group
import datetime

def avg_word_level_bias(word_gradients):
    avg_gradients = {}
    for word in word_gradients[0]:
        word_grads = [seed_grad[word] for seed_grad in word_gradients]
        avg_gradients[word] = [sum(grads)/len(grads) for grads in zip(*word_grads)]
    return avg_gradients

def save_global_word_level_bias(word_gradients, caption, caption_id, question, classes, file_path):
    avg_gradients = avg_word_level_bias(word_gradients)
    
    save_word_level_bias(
        avg_gradients,
        caption,
        caption_id,
        question,
        classes,
        file_path
    )

def build_choices(templates, bias_group, classes):
    choices = []
    for choice_template in templates:
        if choice_template.count('{}') == 2:
            choices.append([choice_template.format(c, bias_group) for c in classes])
        else:
            choices.append([c for c in classes])
    return choices

def worker(process_id, gpu_ids, opt, splitted_data, classes):
    # Set CUDA_VISIBLE_DEVICES to control which GPUs are visible to this process
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu_id) for gpu_id in gpu_ids)

    devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids]

    print(f"Process {os.getpid()} is using GPUs {','.join(devices)}")

    grad_bias = GradBias(
        gen_config=opt['generator'],
        vqa_info=opt['vqa'],
        devices=devices,
        save_info=opt['save_info'],
        loss_interval=opt['loss_interval']
    )

    progress_bar = tqdm(splitted_data, position=process_id*2, leave=True, desc=f'GPU Node: {process_id}, Biases')
    save_dir = opt['save_path']
    for caption, caption_id, bias_group, bias_name, cluster, question in progress_bar:
        global_word_gradients_max, global_word_gradients_mean, global_word_gradients_min = [], [], []

        # update progress bar description
        progress_bar.set_description(f'GPU Node: {process_id}, Biases: {caption_id}')

        bias_identifier = f'{bias_group}_{bias_name}_{cluster}'
        current_path = os.path.join(save_dir, caption_id, bias_identifier)
        os.makedirs(current_path, exist_ok=True)

        for seed in tqdm(opt['seeds'], position=process_id*2+1, leave=False, desc='Seeds'):
            grad_bias.set_seed(seed)
            seed_path = os.path.join(current_path, f'seed_{seed}')
            os.makedirs(seed_path, exist_ok=True)
            grad_bias.set_path(seed_path)

            choices = build_choices(opt['choices_templates'], bias_group, classes[bias_group][bias_name][cluster]['classes'])

            word_gradients_mean = grad_bias.run_pipeline(
                prompt=caption,
                question=f'{question} Answer with one word',
                choices=choices
            )
            global_word_gradients_mean.append(word_gradients_mean)
            
            # save word scores at seed level (mean of the tokens)
            save_word_level_bias(
                word_gradients_mean,
                caption,
                caption_id,
                question,
                classes[bias_group][bias_name][cluster]['classes'],
                os.path.join(seed_path, 'word_level_bias_mean.txt')
            )

        # save word scores at global level (mean of the seeds)
        save_global_word_level_bias(global_word_gradients_mean, caption, caption_id, question, classes[bias_group][bias_name][cluster]['classes'], os.path.join(current_path, 'avg_word_level_bias_mean.txt'))
        torch.cuda.empty_cache()    

def main():
    opt = arg_parse.argparse_gradBias()
    # Initialize the multiprocessing context
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    num_processes = num_gpus // opt['N_GPU_PER_PROC']  # Number of processes to create

    assert num_gpus%opt['N_GPU_PER_PROC'] == 0, f"num_gpus should be a multiple of {opt['N_GPU_PER_PROC']}"

    proposed_biases = datasets.GradBias_proposed_biases(
        dataset_path = DATASET_CONFIG[opt['dataset']]['path'],
        max_prompts = DATASET_CONFIG[opt['dataset']]['max_prompts_per_bias'],
        filter_threshold = DATASET_CONFIG[opt['dataset']]['filter_threshold'],
        hard_threshold = DATASET_CONFIG[opt['dataset']]['hard_threshold'],
        merge_threshold = DATASET_CONFIG[opt['dataset']]['merge_threshold'],
        valid_bias_fn = DATASET_CONFIG[opt['dataset']]['valid_bias_fn'],
        filter_caption_fn = DATASET_CONFIG[opt['dataset']]['filter_caption_fn'],
        all_images = DATASET_CONFIG[opt['dataset']]['all_images'],
        specifc_biases = BIASES_TO_CHECK
    )
    data = proposed_biases.get_data()
    classes = proposed_biases.get_classes()

    data_undone = []
    save_dir = opt['save_path']
    for caption, caption_id, bias_group, bias_name, cluster, question in data:
        bias_identifier = f'{bias_group}_{bias_name}_{cluster}'
        current_path = os.path.join(save_dir, str(caption_id), bias_identifier)
        if not os.path.exists(current_path) or not (len(os.listdir(current_path))-3) >= len(opt['seeds']):
            data_undone.append((caption, str(caption_id), bias_group, bias_name, cluster, question))

    # Divide the GPUs into groups, each group containing N_GPU_PER_PROC GPUs
    gpu_groups = [list(range(i, min(i + opt['N_GPU_PER_PROC'], num_gpus))) for i in range(0, num_gpus, opt['N_GPU_PER_PROC'])]
    nodes = len(gpu_groups)
    splitted_data = np.array_split(data_undone, nodes)

    # Spawn a process for each GPU group
    processes = []
    for process_idx, gpu_group in enumerate(gpu_groups[:num_processes]):
        p = mp.Process(target=worker, args=(process_idx, gpu_group, opt, splitted_data[process_idx], classes))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()