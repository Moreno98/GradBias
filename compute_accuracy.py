import os
import json
import ast
from utils.config import DATASET_CONFIG, BIASES_TO_CHECK
from utils.datasets import LLM_dataset
import utils.utils as utils
from utils.utils import STOP_WORDS
from utils.generalizer import Generalizer
from tabulate import tabulate

def compute_accuracy(method_ranking, true_words, gt_ranking, k, accuracy_values):
    found = False
    i = 0
    true_words = [word.lower() for word in true_words]
    while not found and i < min(len(gt_ranking), k):
        if method_ranking[i].lower() in true_words:
            for topk in range(i, k):
                accuracy_values[topk] += 1
            found = True
        i += 1
    return accuracy_values

def filter_words(method_ranking, gt_ranking):
    final_ranking = []
    for word in method_ranking:
        word = word.lower()
        if word in gt_ranking:
            final_ranking.append(word)
    return final_ranking

def main():
    dataset = 'coco'
    generator = 'sd-2'
    gt_mode = 'remove'
    vqa_model_name_GT = 'blip2-flant5xxl'
    device = 'cuda:0'
    root_path = 'methods'
    generalizer = Generalizer(device)
    loss = 'matching_loss'
    loss_intervals = ['1']
    vqa_GradBias = 'clip-L'
    LLM_models = ['llama2-13B', 'llama3-8B']
    vqa_baseline_models = ['llava-1.5-13b']
    table_path = 'tables'
    syntax_tree = {
        'syntax_tree_subj_related': 'subj',
    }

    DATASET_CONFIG[dataset]['biases_to_check'] = BIASES_TO_CHECK
    gen_images_path = f'generated_images/{dataset}/{generator}/gt/{gt_mode}'

    bias_classes, captions_text = LLM_dataset(
        dataset_setting = DATASET_CONFIG[dataset],
        save_path=os.path.join(
            'LLM_Baseline',
            dataset,
            'llama2',
        ),
        file_name='Rankings.json'
    ).get_accuracy_info()

    vqa_answers = {}
    with open(f'{root_path}/VQA_gt/{dataset}/{vqa_model_name_GT}/{generator}/gt/{gt_mode}/vqa_answers.json', 'r') as f:
        vqa_answers = json.load(f)

    caption_ids = os.listdir(gen_images_path)

    gt = {}
    gt_eq_scores = {}
    counts = 0
    for caption_id in caption_ids:
        gt[caption_id] = {}
        gt_eq_scores[caption_id] = {}
        words = os.listdir(os.path.join(gen_images_path, caption_id))
        seeds = len(os.listdir(os.path.join(gen_images_path, caption_id, words[0])))
        for bias_name in vqa_answers[os.path.join(gen_images_path, caption_id, words[0], f'0.jpg')]:
            bias_cluster = vqa_answers[os.path.join(gen_images_path, caption_id, words[0], f'0.jpg')][bias_name][0]
            class_cluster = vqa_answers[os.path.join(gen_images_path, caption_id, words[0], f'0.jpg')][bias_name][1]
            bias_keys = f'{bias_cluster}/{bias_name}/{class_cluster}'
            classes = bias_classes[bias_cluster][bias_name][class_cluster]['classes']

            bias_info = bias_name
            if bias_cluster.lower() not in bias_name.lower().split():
                bias_info = f'{bias_cluster} {bias_name}'

            full_prompt_classes = {
                class_: 0 for class_ in classes
            }
            for seed in range(seeds):
                answer = vqa_answers[os.path.join(gen_images_path, caption_id, 'full_prompt', f'{seed}.jpg')][bias_name][2]
                full_prompt_classes[answer] += 1

            candidate_words = words[:]
            candidate_words.remove('full_prompt')

            ranking = []
            for word in candidate_words:
                filtered_word = word.split('_')[0]
                if not utils.is_bias_related(filtered_word, bias_info, generalizer) and not utils.is_stop_word(filtered_word) and utils.is_valid_word(filtered_word) and filtered_word != '\'s':
                    answers = {
                        class_: 0 for class_ in classes
                    }
                    for seed in range(seeds):
                        answer = vqa_answers[os.path.join(gen_images_path, caption_id, word, f'{seed}.jpg')][bias_name][2]
                        answers[answer] += 1
                
                    deltas = 0
                    for cls in classes:
                        deltas += abs(answers[cls] - full_prompt_classes[cls])

                    ranking.append((word.lower(), deltas//2))
            
            # check if at least one word changes the distribution
            if not all([word[1] == 0 for word in ranking]):
                ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
                gt[caption_id][bias_keys] = [word[0] for word in ranking]

                ranking_checked = []
                # resort ranking by same deltas
                i = 0
                gt_pos = 0
                while i < len(ranking):
                    ranking_checked.append([ranking[i]])
                    j = i + 1
                    while j < len(ranking) and ranking[i][1] == ranking[j][1]:
                        ranking_checked[gt_pos].append(ranking[j])
                        j += 1
                    i += len(ranking_checked[gt_pos])
                    gt_pos += 1

                gt_eq_scores[caption_id][bias_keys] = []
                for l in ranking_checked:
                    gt_eq_scores[caption_id][bias_keys].append([word[0] for word in l])

            else:
                counts += 1

    print(f"Same distribution on {counts} prompts")

    path = f'{root_path}/GradBias/{vqa_GradBias}/{dataset}/{generator}/{loss}/'
    GradBias_results = {}
    for loss_interval in loss_intervals:
        GradBias_results[loss_interval] = {}
        for caption_id in gt:
            GradBias_results[loss_interval][caption_id] = {}

    k = 4
    top_k = {
        i: 0 for i in range(k)
    }
    accuracies = {
        'GradBias': {
            loss_interval: top_k.copy() for loss_interval in loss_intervals
        },
    }
    accuracies = {}

    for st in syntax_tree:
        accuracies[st] = top_k.copy()

    for llm in LLM_models:
        accuracies[llm] = top_k.copy()
    
    for vqa_baseline_model in vqa_baseline_models:
        accuracies[vqa_baseline_model] = top_k.copy()
    
    data_points = 0

    syntax_tree_results = {}
    for syntax_tree_approach in syntax_tree:
        with open(f'{root_path}/Syntax_tree_baseline/{dataset}/Rankings_{syntax_tree[syntax_tree_approach]}.json', 'r') as f:
            syntax_tree_results[syntax_tree_approach] = json.load(f)
    
    LLM_results = {}
    for LLM_model in LLM_models:
        with open(f'{root_path}/LLM_Baseline/{dataset}/{LLM_model}/Rankings.json', 'r') as f:
            LLM_results[LLM_model] = json.load(f)

    vqa_baseline_results = {}
    for vqa_baseline_model in vqa_baseline_models:
        with open(f'{root_path}/VQA_baseline/{dataset}/{vqa_baseline_model}/{generator}/{gt_mode}/vqa_answers.json', 'r') as f:
            vqa_baseline_results[vqa_baseline_model] = json.load(f)

    for caption_id in gt:
        for bias_key in gt[caption_id]:
            # GradBias Performance
            for loss_interval in loss_intervals:
                bias_cluster = bias_key.split('/')[0]
                bias_name = bias_key.split('/')[1]
                class_cluster = bias_key.split('/')[2]

                bias_info = bias_name
                if bias_cluster.lower() not in bias_name.lower().split():
                    bias_info = f'{bias_cluster} {bias_name}'

                with open(os.path.join(path, loss_interval, caption_id, '_'.join([bias_cluster, bias_name, class_cluster]), 'avg_word_level_bias_mean.txt'), 'r') as f:
                    data = f.readlines()
                    for idx, l in enumerate(data):
                        if 'Choices' in l:
                            starting_idx = idx + 1
                            break
                    word_level_grad = data[starting_idx:]
                    
                    ranking = []
                    occurrences = {}
                    subj_found = False
                    for word in word_level_grad:
                        word_text = str(word.split(':')[0]).lower().replace('.', '')
                        filtered_word = word_text.split('_')[0].translate(str.maketrans('', '', '!"#$%&()*+,-./:;<=>?@[\\]^{|}~'))
                        if '\'s' in filtered_word:
                            word_text = word_text.replace('\'s', '')
                            filtered_word = filtered_word.replace('\'s', '')
                        if not utils.is_bias_related(filtered_word, bias_info, generalizer) and not utils.is_stop_word(filtered_word) and utils.is_valid_word(filtered_word):
                            if word_text not in occurrences:
                                occurrences[word_text] = 1
                            else:
                                occurrences[word_text] += 1
                                word_text = f"{word_text}_{occurrences[word_text]}"
                            grad_text = word.split(':')[1].strip()
                            try:
                                grad_text = ast.literal_eval(grad_text)
                                word_grad = [float(grad) for grad in grad_text]
                            except Exception as e:
                                print(f"Error {e} in ast.literal_eval: {grad_text}")
                                print(loss_interval, caption_id, bias_key)
                                quit()
                            avg_grad = sum(word_grad)/len(word_grad)
                            # remove punctuation from word_text
                            word_text = word_text.translate(str.maketrans('', '', '!"#$%&\()*+,./:;<=>?@[\\]^{|}~'))
                            ranking.append((word_text, avg_grad))

                    ranking = sorted(ranking, key=lambda x: x[1], reverse=True) 
                    # assert len(ranking) == len(gt[caption_id][bias_key]), f"Length of ranking {len(ranking)} is not equal to length of ground truth {len(gt[caption_id][bias_key])}, {bias_cluster}, {caption_id}, caption: {captions_text[int(caption_id)][0]}. GradBias results: {ranking}, GT: {gt[caption_id][bias_key]}"
                    if set([word[0] for word in ranking]) != set(gt[caption_id][bias_key]):
                        word_ranking_set = set([word[0] for word in ranking])
                        gt_set = set(gt[caption_id][bias_key])
                        # print the different words
                        print(f"Words in ranking but not in GT: {word_ranking_set.difference(gt_set)}")
                        print(f'{word_ranking_set.union(gt_set).difference(word_ranking_set.intersection(gt_set))} - {caption_id} - {bias_key} - {captions_text[int(caption_id)][0]}')

                        # print(f"Length of ranking {len(ranking)} is not equal to length of ground truth {len(gt[caption_id][bias_key])}, {bias_cluster}, {caption_id}, caption: {captions_text[int(caption_id)][0]}. GradBias results: {ranking}, GT: {gt[caption_id][bias_key]}")
                    
                    if len(ranking) == len(gt[caption_id][bias_key]):
                        GradBias_results[loss_interval][caption_id][bias_key] = [word[0] for word in ranking]
    
    for cpt_id in gt:
        for bias_key in gt[cpt_id]:
            gt_ranking = gt[cpt_id][bias_key]
            
            # compute top-1, top-2, top-3, top-4
            true_words = gt_eq_scores[cpt_id][bias_key][0]
            for loss_interval in GradBias_results:
                GradBias_ranking = GradBias_results[loss_interval][cpt_id][bias_key]
                accuracies['GradBias'][loss_interval] = compute_accuracy(GradBias_ranking, true_words, gt_ranking, k, accuracies['GradBias'][loss_interval])

            for syntax_tree_approach in syntax_tree_results:
                if 'root' not in syntax_tree_approach:
                    syntax_tree_ranking = syntax_tree_results[syntax_tree_approach][cpt_id][bias_key]
                else:
                    syntax_tree_ranking = syntax_tree_results[syntax_tree_approach][cpt_id]
                accuracies[syntax_tree_approach] = compute_accuracy(filter_words(syntax_tree_ranking, gt_ranking), true_words, gt_ranking, k, accuracies[syntax_tree_approach])

            for llm_model in LLM_results:
                LLM_ranking = LLM_results[llm_model][cpt_id][bias_key]
                LLM_ranking = [value[0] for value in LLM_ranking]
                accuracies[llm_model] = compute_accuracy(filter_words(LLM_ranking, gt_ranking), true_words, gt_ranking, k, accuracies[llm_model])

            for vqa_baseline_model in vqa_baseline_results:
                vqa_baseline_ranking = vqa_baseline_results[vqa_baseline_model][cpt_id][bias_key]
                vqa_baseline_ranking = [value[0] for value in vqa_baseline_ranking]
                accuracies[vqa_baseline_model] = compute_accuracy(vqa_baseline_ranking, true_words, gt_ranking, k, accuracies[vqa_baseline_model])
            data_points += 1

    for model in accuracies:
        if model == 'GradBias':
            for loss_interval in accuracies[model]:
                for topk in accuracies[model][loss_interval]:
                    accuracies[model][loss_interval][topk] = round(accuracies[model][loss_interval][topk]/data_points, 4)
        else:
            for topk in accuracies[model]:
                accuracies[model][topk] = round(accuracies[model][topk]/data_points, 4)

    for method in accuracies:
        if method == 'GradBias':
            for loss_interval in accuracies[method]:
                print(f"GradBias {loss_interval} performance: {accuracies[method][loss_interval]}")
        else:
            print(f"{method} performance: {accuracies[method]}")

    # use tabulate to print table in latex format
    table = []
    table.append(['Model'] + [f'Top-{i+1}' for i in range(k)])
    for model in accuracies:
        if model == 'GradBias':
            for loss_interval in accuracies[model]:
                table.append([f'{model} {loss_interval}'] + [accuracies[model][loss_interval][i] for i in range(k)])
        else:
            table.append([model] + [accuracies[model][i] for i in range(k)])
    
    os.makedirs(table_path, exist_ok=True)
    with open(f'{table_path}/{generator}.txt', 'w+') as f:
        f.write(tabulate(table, tablefmt='latex_raw'))
        
if __name__ == '__main__':
    main()
