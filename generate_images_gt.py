import utils.arg_parse as arg_parse
from utils.config import DATASET_CONFIG, BIASES_TO_CHECK
import utils.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
from utils.DDP_manager import DDP  
from torch.utils.data import Dataset
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import utils.utils as utils
from utils.generalizer import Generalizer
from utils.generative_models import Stable_Diffusion, Stable_Diffusion_XL
import random
import sys

class Distributed_dataset(Dataset):
    def __init__(self, 
            rank, 
            world_size, 
            opt,
            ds
        ):
        self.rank = rank
        self.world_size = world_size
        data = ds.get_data()
        self.data_to_generate = []
        for bias_cluster, prompt, caption_id in data:
            self.data_to_generate.append((bias_cluster, prompt, caption_id))

        length = len(self.data_to_generate)
        samples_per_rank = length // world_size
        if rank == world_size-1:
            self.data_to_generate = self.data_to_generate[rank*samples_per_rank:]
        else:
            self.data_to_generate = self.data_to_generate[rank*samples_per_rank: (rank+1)*samples_per_rank]
    def __getitem__(self, idx):
        bias_cluster, caption, caption_id = self.data_to_generate[idx]
        return bias_cluster, caption, caption_id
    def __len__(self):
        return len(self.data_to_generate)

class DDP_Generation(DDP):
    def __init__(
        self, 
        rank, 
        world_size,
        opt,
        ds
    ):
        self.opt = opt
        self.ds = ds
        self.generalizer = None
        self.generalizer = Generalizer(device=f'cuda:{rank}')
        super(DDP_Generation, self).__init__(rank, world_size)
        self.generalizer.quit()

    def generate(self, prompt, seeds):
        prompts = [prompt+' '+self.opt['generator']['pos_prompt']]
        return self.generative_model.generate_images(prompt=prompts, seeds=seeds)

    def main(self):
        self.generative_model = eval(self.opt['generator']['class'])(
            gen_config=self.opt['generator'], 
            device=self.device, 
            n_images=self.opt['n-images'],
            inference_steps=50
        )
        ds = Distributed_dataset(
            rank = self.rank,
            world_size = self.world_size,
            opt = self.opt,
            ds = self.ds
        )
        
        print(f'Rank {self.rank} has {len(ds)} samples to generate')
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        for bias_cluster, prompts, caption_ids in tqdm(loader, position=self.rank*2, desc=f'Rank {self.rank}', leave=True):
            full_prompt = prompts[0]
            final_prompts = [(full_prompt, 'full_prompt', 0)]
            
            spacy_doc = utils.spacy_nlp(full_prompt, keep_dash=True)
            filtered_tokens, tested_tokens_counts = utils.filter_stopwords_tokens(spacy_doc)
            for tested_token in filtered_tokens:
                # if tested_token.text.lower() not in subj:
                if not utils.is_bias_related(tested_token.text, bias_cluster[0], self.generalizer):
                    # if token dep is Compound modifier (compound), adjectival modifier (amod) or adverbial modifier (advmod), remove the word (https://github.com/clir/clearnlp-guidelines/blob/master/md/specifications/dependency_labels.md)
                    if tested_token.dep_ in ['compound', 'amod', 'advmod']:
                        updated_word = ''
                    else:
                        updated_word = self.opt['edit_word'](token=tested_token, generalizer=self.generalizer)
                        # if the word is none, remove it
                        updated_word = updated_word if updated_word is not None else ''

                    edited_prompt = utils.replace(full_prompt, tested_token.text, updated_word, tested_tokens_counts[tested_token.text])
                    final_prompts.append((edited_prompt, tested_token.text, tested_tokens_counts[tested_token.text]))
                    tested_tokens_counts[tested_token.text] += 1

            for prompt, tested_word, count in tqdm(final_prompts, position=self.rank*2+1, desc=f'Rank {self.rank} - {caption_ids[0]}', leave=False):
                caption_id = str(caption_ids[0].item()) if type(caption_ids[0]) == torch.Tensor else str(caption_ids[0])
                word_identifier = tested_word
                if count > 1:
                    word_identifier = tested_word + f'_{count}'
                # check if the images are already generated
                if os.path.isdir(os.path.join(self.opt['save_path'], caption_id, word_identifier)):
                    if len(os.listdir(os.path.join(self.opt['save_path'], caption_id, word_identifier))) >= len(self.opt['seeds']):
                        continue
                # GENERATE 
                gen_images = self.generate(prompt, seeds=self.opt['seeds'])
                save_dir = os.path.join(self.opt['save_path'], caption_id, word_identifier)
                os.makedirs(save_dir, exist_ok=True)
                for image_idx, image in enumerate(gen_images):
                    image.save(os.path.join(save_dir, f'{image_idx}.jpg'))
                    if not os.path.isfile(os.path.join(save_dir, f'{image_idx}.jpg')):
                        print(f'ERROR: image {image_idx} of caption {caption_ids[0]} not saved')
             

def run(rank, world_size, opt, ds):
    DDP_Generation(
        rank = rank,
        world_size = world_size,
        opt = opt,
        ds = ds
    )

def main():
    opt = arg_parse.argparse_image_gen_gt()
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} GPUs')
    mp.set_start_method('spawn')

    proposed_biases = datasets.Proposed_biases(
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

    mp.spawn(
        run, 
        args=(world_size, opt, proposed_biases), 
        nprocs=world_size
    )

if __name__ == '__main__':
    main()