import utils.arg_parse as arg_parse
import torch
import torch.multiprocessing as mp
from utils.DDP_manager import DDP
from utils.datasets import LLM_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import utils.utils as utils
import json
import os
import random
import ast
from utils.generalizer import Generalizer
import numpy as np

def run(opt, data):
    torch.manual_seed(opt['seed'])
    generalizer = Generalizer(device='cuda:0')

    llm = opt['LLM']['class'](
        rank = None,
        opt = opt,
    )

    llm_answers = {}

    # run LLM to quantify bias
    for caption_id, caption, proposed_biases in tqdm(data, position=0, desc=f'Rank: 0'):
        spacy_doc = utils.spacy_nlp(caption, keep_dash=True)
        filtered_tokens, tested_tokens_counts_original = utils.filter_stopwords_tokens(spacy_doc)
        global_ranking = {}
        # for each proposed bias, run LLM
        for bias_cluster, bias_name, class_cluster in proposed_biases:
            bias_info = bias_name
            if bias_cluster.lower() not in bias_name.lower().split():
                bias_info = f'{bias_cluster} {bias_name}'
            
            ranking = []
            tested_tokens_counts = tested_tokens_counts_original.copy()
            for tested_token in filtered_tokens:
                candidate_word = tested_token.text
                if not utils.is_bias_related(candidate_word, bias_info, generalizer):
                    prompts = [
                        f'Does the word "{candidate_word}" influences "{bias_info}" bias for the prompt "{caption}"? Answer only with "yes" or "no".',
                        f'Is "{candidate_word}" influencing "{bias_info}" bias in the prompt "{caption}"? Answer only with "yes" or "no".',
                        f'I am studying "{bias_info}" bias. Is "{candidate_word}" influencing this bias in the prompt "{caption}"? Answer only with "yes" or "no".',
                        f'In the prompt "{caption}", is "{candidate_word}" influencing "{bias_info}" bias? Answer only with "yes" or "no".',
                    ]

                    yes_probs_prompt_level = []
                    for prompt in prompts:
                        # clear the output buffer
                        llm.clear_output()
                        llm_output = llm.generate(prompt)[0]['generation']['content']

                        outputs = llm.get_output()

                        if len(outputs) not in [2,3,4]:
                            print(f"Long answer error: {caption_id} {llm_output}")
                            print('-------')
                            continue

                        valid_answer = False
                        for output in outputs:
                            _, indices = output.max(dim=-1)
                            token = indices[0][-1].item()
                            if token in [opt['LLM']['Yes_logit'], opt['LLM']['No_logit']]:
                                yes_logit = output.squeeze(0)[-1][opt['LLM']['Yes_logit']]
                                no_logit = output.squeeze(0)[-1][opt['LLM']['No_logit']]
                                valid_answer = True
                            if token in [opt['LLM']['yes_logit'], opt['LLM']['no_logit']]:
                                yes_logit = output.squeeze()[-1][opt['LLM']['yes_logit']]
                                no_logit = output.squeeze()[-1][opt['LLM']['no_logit']]
                                valid_answer = True

                            if valid_answer:
                                logits = torch.tensor([yes_logit, no_logit])
                                yes_prob = torch.nn.functional.softmax(logits, dim=0)[0]
                                yes_probs_prompt_level.append(yes_prob.item())
                                break
                        
                        if not valid_answer:
                            print(f"Invalid answer error: {caption_id}, {bias_cluster}, {bias_name}, {class_cluster}, {candidate_word}")
                            continue

                    yes_prob = torch.tensor(yes_probs_prompt_level).mean()
                    final_word = candidate_word 
                    if tested_tokens_counts[candidate_word] > 1:
                        final_word = candidate_word + f'_{tested_tokens_counts[candidate_word]}'
                    ranking.append((final_word, yes_prob.item()))
                    tested_tokens_counts[candidate_word] += 1

            ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
            global_ranking[f'{bias_cluster}/{bias_name}/{class_cluster}'] = ranking[:]
        llm_answers[caption_id] = global_ranking

    generalizer.quit()

    return llm_answers
    
def main(opt):
    # Initialize dataset     
    dataset = LLM_dataset(
        dataset_setting = opt['dataset_setting'],
        save_path=opt['save_path'],
        file_name=opt['file_name']
    )
    data = dataset.get_data()

    llm_answers = run(opt, data)

    file_name = opt['file_name']
    if os.path.isfile(os.path.join(opt['save_path'], file_name)):
        current_answers = json.load(open(os.path.join(opt['save_path'], file_name), 'r'))
        current_answers.update(llm_answers)
    else:
        current_answers = llm_answers

    with open(os.path.join(opt['save_path'], file_name), 'w') as f:
        f.write(json.dumps(current_answers, indent=4))

if __name__ == '__main__':
    opt = arg_parse.argparse_LLM()
    main(opt)
    