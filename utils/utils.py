from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
import spacy
import string
import torch
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import psutil

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
# add punctuation to stopwords
STOP_WORDS.update(set(string.punctuation))
nlp = spacy.load("en_core_web_sm")

TOTAL_GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory/1024**2

def filter_stopwords_tokens(spacy_doc):
    filtered_tokens = []
    for token in spacy_doc:
        if token.text.lower() not in STOP_WORDS:
            filtered_tokens.append(token)

    token_counts = {}
    for token in filtered_tokens:
        token_counts[token.text] = 1
    
    return filtered_tokens, token_counts

# BFS from the root token
def BFS(root):
    queue = [root]
    visited = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(node.children)
    return visited

def clear_cuda_memory(ranks):
    # get available gpu memory
    for rank in ranks:
        r = torch.cuda.memory_reserved(rank)/1024**2+1500
        if TOTAL_GPU_MEMORY-r < 2000:
            torch.cuda.empty_cache()

def get_subject(text):
    doc = nlp(text)
    subjects = []
    for token in doc:
        if is_subj(token):
            subjects.append(token.text)
    
    if len(subjects) == 0:
        # Check if the root of the dependency tree is a noun phrase
        root = get_root_token(text)    
        if root is not None and root.pos_ == 'NOUN':
            return [root.text]
        else:
            return ['']
    else:
        return subjects

def get_root_token(text):
    doc = spacy_nlp(text, keep_dash=True)
    return list(doc.sents)[0].root

def syntax_tree_baseline(root_token):
    visited = BFS(root_token)
    # remove stopwords
    return [t for t in visited if t.text.lower() not in STOP_WORDS]    

def filter_stopwords(text):
    # remove punctuation first
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word.lower() not in STOP_WORDS])
    
def is_stop_word(word):
    return word.lower() in STOP_WORDS

def is_valid_word(word):
    return word != ' ' and word != '' and word != '\n' and word != '\t'

def replace(sentence, old_word, new_word, occurrance):
    assert occurrance > 0, "occurrance should be greater than 0"
    # counts occurances of old_word in sentence
    assert sentence.count(old_word) >= occurrance, f"occurrance should be less than or equal to the number of times {old_word} appears in the sentence"
    return old_word.join(sentence.split(old_word)[:occurrance]) + new_word + old_word.join(sentence.split(old_word)[occurrance:])

def setup_gpu(devices, CLIP = False):
    n_gpu = len(devices)
    if n_gpu == 4 or n_gpu == 3:
        sd_devices = devices[-1]
        vqa_devices = devices[:-1]
        ranks = [int(device[-1]) for device in vqa_devices] + [int(sd_devices[-1])]
    elif n_gpu == 2:
        sd_devices = devices[-1]
        if not CLIP:
            vqa_devices = devices
        else:
            vqa_devices = devices[:-1]
        ranks = [int(device[-1]) for device in vqa_devices] + [int(sd_devices[-1])]
    elif n_gpu == 1:
        sd_devices = devices[0]
        vqa_devices = [devices[0]]
        ranks = [int(sd_devices[-1])]
 
    return sd_devices, vqa_devices, ranks

def get_max_memory(devices):
    max_memory = {int(device[-1]): torch.cuda.mem_get_info(device)[0] for device in devices}
    max_memory['cpu'] = psutil.virtual_memory().available
    return max_memory

def spacy_nlp(text, keep_dash=False):
    if keep_dash:
        inf = list(nlp.Defaults.infixes)
        inf = [x for x in inf if '-|–|—|--|---|——|~' not in x] # remove the hyphen-between-letters pattern from infix patterns
        infix_re = compile_infix_regex(tuple(inf))
        nlp.tokenizer = Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                            suffix_search=nlp.tokenizer.suffix_search,
                            infix_finditer=infix_re.finditer,
                            token_match=nlp.tokenizer.token_match,
                            rules=nlp.Defaults.tokenizer_exceptions)
    return nlp(text)

def is_subj(token):
    return token.dep_ in ['nsubj', 'nsubjpass', 'csubj', 'attr']

def generic_word(token, generalizer):
    return generalizer.generalize(token.text)

def remove_word(**kwargs):
    return ''

def save_word_level_bias(word_gradients, prompt, prompt_id, question, classes, path):
    with open(path, 'w') as f:
        f.write(f'Prompt: {prompt}\n')
        f.write(f'Prompt ID: {prompt_id}\n')
        f.write(f'Question: {question}\n')
        f.write(f'Choices: {classes}\n')
        for word, grads in word_gradients.items():
            f.write(f'{word}: {grads}\n')

def is_bias_related(word, bias, generalizer):
    bias = bias.split()
    if word.lower() in bias:
        return True
    for b in bias:
        generalized_word = generalizer.generalize(word)
        if generalized_word == b:
            return True
        generalized_bias = generalizer.generalize(b)
        if generalized_bias == word or generalized_bias == generalized_word:
            return True
    return False

