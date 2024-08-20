# GradBias
## Installation
We recomand to use a virtual environment to install the required environment.
```bash
# Create a virtual environment and activate it
python -m venv gradbias
source gradbias/bin/activate
```
Before installing the required packages, please install [PyTorch](https://pytorch.org/get-started/locally/) separately according to your system and CUDA version.  
After installing PyTorch, you may install the required packages with the following commands:
```bash
# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```
Please, install a spaCy trained pipeline with the following command:
```bash
python -m spacy download en_core_web_sm
```
This code has been tested with `PyTorch 2.2.1`, `CUDA 11.8`, `CUDA 12.1` and `python 3.10.9`.

## Usage
We provide code to:
- Run the introduced baselines on OpenBias extracted biases.
- Run GradBias on the same dataset.
- Run GradBias indipendently with custom prompts and biases.
We make available the dataset used in our experiments here: [Dataset](https://drive.google.com/file/d/1nGECdt0fcwiJA-5qJgvgnZbGBp4zHnNq/view?usp=sharing). This file should be put under `proposed_biases/coco/3` folder.

### Baselines
To run the baselines, you can use the following commands:
```bash
# Run syntax tree baseline
...
# Run LLM baseline
...
# Run VQA baseline
...
```
The results will be saved in the `results` folder.

### GradBias
...

