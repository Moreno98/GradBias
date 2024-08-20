from utils.VQA import SBERTModel
from nltk.corpus import wordnet
import nltk
import os
import json
from filelock import FileLock

nltk.download('wordnet')

class Generalizer:
    def __init__(self, device='cuda:0') -> None:
        self.category_file_path = os.path.join("data", "categories.json")
        self.categories = {}
        self.sbert_model = None
        self.device = device
        if os.path.exists(self.category_file_path):
            self.save = False
            with open(self.category_file_path, "r") as f:
                self.categories = dict(json.load(f))
        else:
            with open(self.category_file_path, "w+") as f:
                json.dump({}, f)
            self.load_sbert_model()

    def get_hypernyms(self, word):
        total_hypernyms = []
        # Get synsets for the given word
        synsets = wordnet.synsets(word)

        # Check if any synsets are found
        if synsets:
            # Get the most general hypernym for the first synset
            hypernym = synsets[0]
            while True:
                hypernyms = hypernym.hypernyms()
                if not hypernyms:
                    break
                hypernym = hypernyms[0]  # Take the first hypernym
                total_hypernyms.append(hypernym.name().split('.')[0])
            return total_hypernyms
        else:
            return None

    def load_sbert_model(self):
        if not self.sbert_model:
            self.sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2", device=self.device)

    def quit(self):
        if self.save:
            with FileLock(self.category_file_path + ".lock"):
                # read file
                with open(self.category_file_path, "r") as f:
                    current_categories = dict(json.load(f))
                # update categories
                new_keys = set(self.categories.keys()) - set(current_categories.keys())
                for key in new_keys:
                    current_categories[key] = self.categories[key]
                # write file
                with open(self.category_file_path, "w+") as f:
                    json.dump(current_categories, f, indent=4)

    def generalize(self, word):
        if word in self.categories:
            return self.categories[word]
        else:
            self.save = True
            self.load_sbert_model()
            hypernyms = self.get_hypernyms(word)
            if hypernyms:
                generic_word = self.sbert_model.multiple_choice(word, hypernyms)
                generic_word = generic_word.replace("_", " ").lower()
                self.categories[word] = generic_word
                return generic_word
            else:
                self.categories[word] = None
                return None 
