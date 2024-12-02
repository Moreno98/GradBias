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
        super(DDP_VQA, self).__init__(rank, world_size)

    def main(self):
        # Initialize VQA model
        vqa_model = VQA(self.device, self.opt)   
  
        # Initialize dataset     
        dataset = VQA_dataset(
            dataset_setting = self.opt['dataset_setting'],
            save_path=self.opt['save_path'],
            file_name=self.opt['file_name'],
            multi_image=False
        )
        loader = DataLoader(
            dataset, 
            batch_size=None, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True, 
            sampler=DistributedSampler(dataset, shuffle=False)
        )

        # run VQA to quantify bias
        for caption_id, caption, image_id, image_path, image, detected_biases in tqdm(loader, position=self.rank, desc=f'Rank {self.rank}'):
            answers = {}
            image = vqa_model.process_image(image)
            # for each detected bias, run VQA
            for bias_cluster, bias_name, class_cluster, question, classes in detected_biases:
                # add UNK class
                # classes.append(self.opt['UNK_CLASS'])
                # run VQA
                answer = vqa_model.get_answer(image, question, choices=classes)
                # get VQA prediction
                class_pred = answer['multiple_choice_answer']
                # update answers
                answers[bias_name] = (
                    bias_cluster,
                    class_cluster,
                    class_pred,
                )
            # update vqa answers
            self.vqa_answers[image_path] = answers

def run(rank, world_size, vqa_answers, opt):
    torch.manual_seed(opt['seed'])
    DDP_VQA(rank, world_size, vqa_answers, opt)

def init_answers(manager, data, opt):
    vqa_answers = manager.dict()
    for caption_id, caption, image_id, image_path, detected_biases in data:
        vqa_answers[image_path] = manager.dict()
    return vqa_answers

def deserialize_answers(vqa_answers):
    vqa_answers = dict(vqa_answers.copy())
    for caption_id in vqa_answers:
        vqa_answers[caption_id] = dict(vqa_answers[caption_id].copy())
    return vqa_answers
    
def main(opt):
    print(f"Initialize MULTI GPUs on {torch.cuda.device_count()} devices")
    world_size = torch.cuda.device_count()
    manager = mp.Manager()

    # Initialize dataset     
    dataset = VQA_dataset(
        dataset_setting = opt['dataset_setting'],
        save_path=opt['save_path'],
        file_name=opt['file_name']
    )

    vqa_answers = init_answers(manager, dataset.get_data(), opt)

    mp.spawn(run, args=(
                        world_size,
                        vqa_answers,
                        opt
                    ), nprocs=world_size)

    vqa_answers = deserialize_answers(vqa_answers)

    # save VQA answers
    file_name = opt['file_name']
    if os.path.isfile(os.path.join(opt['save_path'], file_name)):
        current_answers = json.load(open(os.path.join(opt['save_path'], file_name), 'r'))
        current_answers.update(vqa_answers)
    else:
        current_answers = vqa_answers
        
    with open(os.path.join(opt['save_path'], file_name), 'w') as f:
        f.write(json.dumps(current_answers, indent=4))

if __name__ == '__main__':
    opt = arg_parse.argparse_VQA_gt()
    main(opt)
    