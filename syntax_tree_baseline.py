import utils.arg_parse as arg_parse
from utils.datasets import LLM_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils
import json
import os
from utils.generalizer import Generalizer

def compute_distance(token, current_distance, visited):
    if token not in visited:
        visited[token] = current_distance
        for child in token.children:
            compute_distance(child, current_distance+1, visited)
        if token.head is not None:
            compute_distance(token.head, current_distance+1, visited)
    
def main(opt):
    answers = {}

    # Initialize dataset     
    dataset = LLM_dataset(
        dataset_setting=opt['dataset_setting'],
        save_path=opt['save_path'],
        file_name=opt['file_name']
    )
    loader = DataLoader(
        dataset, 
        batch_size=None, 
        shuffle=False, 
        num_workers=4, 
    )

    generalizer = Generalizer(device='cuda:0')

    # run VQA to quantify bias
    for caption_id, caption, proposed_biases in tqdm(loader):
        doc = utils.spacy_nlp(caption, keep_dash=True)
        filtered_tokens, tested_tokens_counts_original = utils.filter_stopwords_tokens(doc)
        answers[caption_id] = {}

        for bias_cluster, bias_name, class_cluster in proposed_biases:
            # this is needed to keep track of multiple occurrences of the same word
            root_token = list(doc.sents)[0].root
            tested_tokens_counts = tested_tokens_counts_original.copy()
            token_associations = {}
            for token in filtered_tokens:
                if tested_tokens_counts[token.text] > 1:
                    token_associations[token] = f'{token.text}_{tested_tokens_counts[token.text]}'
                else:
                    token_associations[token] = token.text
                tested_tokens_counts[token.text] += 1  

            filtered_tokens_text = [token.text for token in filtered_tokens]

            bias_info = bias_name
            if bias_cluster.lower() not in bias_name.lower().split():
                bias_info = f'{bias_cluster} {bias_name}'

            subj_token = None
            for token in filtered_tokens:
                if utils.is_subj(token):
                    subj_token = token
                    break
            
            if subj_token is None:
                # if no subject is found, run BFS from the root token
                ranking_sorted = utils.syntax_tree_baseline(root_token)
            else:
                scores = {}
                compute_distance(subj_token, 0, scores) # compute distance from the subject -- i.e., BFS from the subject                         
                ranking_sorted = sorted(scores, key=scores.get)

            ranking = []
            for token in ranking_sorted:
                if token.text in filtered_tokens_text and not utils.is_bias_related(token.text, bias_info, generalizer):
                    try:
                        token_text = token_associations[token]
                    except KeyError:
                        print('error')
                    ranking.append(token_text)

            answers[caption_id][f'{bias_cluster}/{bias_name}/{class_cluster}'] = ranking[:]

    with open(os.path.join(opt['save_path'], opt['file_name']), 'w') as f:
        f.write(json.dumps(answers, indent=4))

    generalizer.quit()

if __name__ == '__main__':
    opt = arg_parse.argparse_sintax_tree()
    main(opt)
    
    