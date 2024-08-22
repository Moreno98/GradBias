import utils.arg_parse as arg_parse
import torch
import torch.multiprocessing as mp
from utils.DDP_manager import DDP
from utils.VQA import VQA
from utils.datasets import VQA_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
import json
import os
import utils.utils as utils
from utils.generalizer import Generalizer
import numpy as np

class DDP_VQA(DDP):
    def __init__(
        self, 
        rank,
        world_size,
        vqa_answers,
        opt
    ):
        self.opt = opt
        self.vqa_answers = vqa_answers
        self.outputs = []
        super(DDP_VQA, self).__init__(rank, world_size)

    def save_output(self, module, input, output):
        self.outputs.append(output)

    def get_output(self):
        return self.outputs

    def clear_output(self):
        self.outputs = []

    def main(self):
        # Initialize VQA model
        vqa_model = VQA(self.device, self.opt, sbert=False)   
  
        vqa_model.model.model.lm_head.register_forward_hook(self.save_output)

        # Initialize dataset     
        dataset = VQA_dataset(
            dataset_setting = self.opt['dataset_setting'],
            save_path=self.opt['save_path'],
            file_name=self.opt['file_name'],
            multi_image=True
        )
        loader = DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True, 
            sampler=DistributedSampler(dataset, shuffle=False)
        )

        vqa_logit_info = self.opt['vqa_model'][2]

        generalizer = Generalizer(device='cuda:0')

        # run VQA to quantify bias
        for caption_id, caption, image_id, general_image_path, loaded_images, candidate_words, detected_biases in tqdm(loader, position=self.rank*2, desc=f'Rank {self.rank}', leave=True):
            spacy_doc = utils.spacy_nlp(caption, keep_dash=True)
            filtered_tokens, tested_tokens_counts_original = utils.filter_stopwords_tokens(spacy_doc)
            caption_level_ranking = {}
            # for each detected bias, run VQA
            for bias_cluster, bias_name, class_cluster, question, classes in detected_biases:
                bias_info = bias_name
                if bias_cluster.lower() not in bias_name.lower().split():
                    bias_info = f'{bias_cluster} {bias_name}'
                    
                ranking = []
                tested_tokens_counts = tested_tokens_counts_original.copy()
                for candidate_word in tqdm(filtered_tokens, position=self.rank*2+1, desc=f'Candidate word', leave=False):
                    candidate_word = candidate_word.text
                    if not utils.is_bias_related(candidate_word, bias_cluster, generalizer):
                        image_path = os.path.join(general_image_path, candidate_word)
                        image_names = os.listdir(image_path)
                        # run VQA
                        prompts = [
                            f'Does the word "{candidate_word}" influences "{bias_info}" bias for the prompt "{caption}"? Anser with "yes" or "no".',
                            f'Is "{candidate_word}" influencing "{bias_info}" bias in the prompt "{caption}"? Anser with "yes" or "no".',
                            f'I am studying "{bias_info}" bias. Is "{candidate_word}" influencing this bias in the prompt "{caption}"? Anser with "yes" or "no".',
                            f'In the prompt "{caption}", is "{candidate_word}" influencing "{bias_info}" bias? Anser with "yes" or "no".',
                        ]
                        
                        yes_probs_prompt_level = []
                        for prompt in prompts:
                            yes_probs = []
                            for image_name in image_names:
                                image = loaded_images[os.path.join(image_path, image_name)]
                                processed_image = vqa_model.process_image(image).to(self.device)
                                
                                self.clear_output()
                                answer = vqa_model.get_answer(processed_image, prompt)

                                outputs = self.get_output()

                                if len(outputs) != 2 and len(outputs) != 3:
                                    print(f"Long answer error: {caption_id}, {bias_cluster}, {bias_name}, {class_cluster}, {candidate_word}")
                                    print('-------')
                                    continue

                                valid_answer = False
                                for output in outputs:
                                    _, indices = output.max(dim=-1)
                                    token = indices[0][-1].item()
                                    if token in [vqa_logit_info['Yes_logit'], vqa_logit_info['No_logit']]:
                                        yes_logit = output.squeeze(0)[-1][vqa_logit_info['Yes_logit']]
                                        no_logit = output.squeeze(0)[-1][vqa_logit_info['No_logit']]
                                        valid_answer = True
                                    if token in [vqa_logit_info['yes_logit'], vqa_logit_info['no_logit']]:
                                        yes_logit = output.squeeze()[-1][vqa_logit_info['yes_logit']]
                                        no_logit = output.squeeze()[-1][vqa_logit_info['no_logit']]
                                        valid_answer = True

                                    if valid_answer:
                                        logits = torch.tensor([yes_logit, no_logit])
                                        yes_prob = torch.nn.functional.softmax(logits.to(torch.float32), dim=0)[0]
                                        yes_probs.append(yes_prob.item())                 
                                        break
                                
                                if not valid_answer:
                                    print(f"Invalid answer error: {caption_id}, {bias_cluster}, {bias_name}, {class_cluster}, {candidate_word}")
                                    continue

                            yes_probs_prompt_level.append(np.mean(yes_probs))

                        avg_yes_prob = np.mean(yes_probs_prompt_level)
                        final_word = candidate_word
                        if tested_tokens_counts[candidate_word] > 1:
                            final_word = candidate_word + f'_{tested_tokens_counts[candidate_word]}'
                        ranking.append((final_word, avg_yes_prob))
                        tested_tokens_counts[candidate_word] += 1

                ranking = sorted(ranking, key=lambda x: x[1], reverse=True)

                caption_level_ranking[f'{bias_cluster}/{bias_name}/{class_cluster}'] = ranking[:]
            # update vqa answers
            self.vqa_answers[caption_id] = caption_level_ranking

        generalizer.quit()

def run(rank, world_size, vqa_answers, opt):
    torch.manual_seed(opt['seed'])
    DDP_VQA(rank, world_size, vqa_answers, opt)

def init_answers(manager, data, opt):
    vqa_answers = manager.dict()
    for caption_id, caption, image_id, image_path, candidate_words, detected_biases in data:
        vqa_answers[caption_id] = manager.dict()
    return vqa_answers

def deserialize_answers(vqa_answers):
    vqa_answers = dict(vqa_answers.copy())
    for caption_id in vqa_answers:
        vqa_answers[caption_id] = dict(vqa_answers[caption_id].copy())
    return vqa_answers
    
def main(opt):
    print(f"Initialize MULTI GPUs on {torch.cuda.device_count()} devices")
    mp.set_start_method('spawn')
    world_size = torch.cuda.device_count()
    manager = mp.Manager()

    # Initialize dataset     
    dataset = VQA_dataset(
        dataset_setting = opt['dataset_setting'],
        save_path=opt['save_path'],
        file_name=opt['file_name'],
        multi_image=True
    )

    vqa_answers = init_answers(manager, dataset.get_data(), opt)

    mp.spawn(run, args=(
                        world_size,
                        vqa_answers,
                        opt
                    ), nprocs=world_size)

    vqa_answers = deserialize_answers(vqa_answers)

    # save VQA answers
    answers = json.dumps(vqa_answers, indent=4)
    file_name = opt['file_name']
    if os.path.isfile(os.path.join(opt['save_path'], file_name)):
        current_answers = json.load(open(os.path.join(opt['save_path'], file_name), 'r'))
        current_answers.update(vqa_answers)
    else:
        current_answers = vqa_answers
        
    with open(os.path.join(opt['save_path'], file_name), 'w') as f:
        f.write(json.dumps(current_answers, indent=4))

if __name__ == '__main__':
    opt = arg_parse.argparse_VQA_baseline()
    main(opt)
    