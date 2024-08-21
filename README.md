# GradBias
## Installation
We recomand to use a virtual environment to install the required environment.
```bash
# Create a virtual environment, activate it and upgrade pip
python -m venv gradbias
source gradbias/bin/activate
pip install --upgrade pip
```
Before installing the required packages, please install [PyTorch](https://pytorch.org/get-started/locally/) separately according to your system and CUDA version.  
After installing PyTorch, you may install the required packages with the following commands:
```bash
#  Install requirements
pip install -r requirements.txt
```
Please, install a spaCy trained pipeline. You may use the following command:
```bash
python -m spacy download en_core_web_sm
```
This code has been tested with `PyTorch 2.2.1`, `CUDA 11.8`, `CUDA 12.1` and `python 3.10.9`.

## Usage
We provide code to:
- Run the introduced baselines on OpenBias extracted biases.
- Run GradBias on the same dataset.
- Run GradBias indipendently with custom prompts and biases.

We make available:
- The dataset used in our experiments: [Dataset](https://drive.google.com/file/d/1nGECdt0fcwiJA-5qJgvgnZbGBp4zHnNq/view?usp=sharing). This file should be put under `proposed_biases/coco/3`. 
- The synonym file: [Synonyms](https://drive.google.com/file/d/1cXWzktkTLVc7ZYw93Ei_YokI8gOd5_0z/view?usp=sharing). This file can be downloaded and put under `data/`, otherwise it will be automatically generated by the code (it may take a while).

The results of the experiments will be saved under the `methods` folder.
### Baselines
Before running the baselines, make sure to download the models weights (e.g., Llama2-7B, Llama2-13B, etc.) using the official repo and update the paths in `utils/config.py` file. 

To run the baselines, you can use the following commands:
```bash
# Run syntax tree baseline
CUDA_VISIBLE_DEVICES=0 python syntax_tree_baseline.py 
# Run LLM baseline
...
# Run VQA baseline
...
```
The results will be saved in the `results` folder.

### GradBias
...

