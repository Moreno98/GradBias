# llava model imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from lavis.models import load_model_and_preprocess
from transformers import AutoProcessor
from transformers import BlipForQuestionAnswering
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import clip
import utils.utils as utils


class VQA():
    def __init__(
        self,
        device,
        opt,
        sbert = False
    ):
        print(f'Loading VQA model: {opt["vqa_model"]}')
        class_name, self.model_path = opt['vqa_model'][0], opt['vqa_model'][1]
        vqa_device = device
        if class_name == 'Llava' and not isinstance(device, list):
            vqa_device = [device]

        self.model = eval(class_name)(device=vqa_device, ckpt=self.model_path, custom_sys=opt['custom_sys'])
        print(f'VQA model loaded')
        self.sbert = sbert
        if sbert:
            print(f'Loading SBERT model')
            self.sbert_model = SBERTModel("sentence-transformers/all-mpnet-base-v2", device=device)
            print(f'SBERT model loaded')
    
    @torch.no_grad()
    def get_caption(self, image, question):
        # Get VQA model's answer
        caption = self.model.vqa(
            image=image, 
            question=question
        )
        return caption

    @torch.no_grad()
    def process_image(self, image):
        image = image.convert('RGB')
        return self.model.process_image(image)

    @torch.no_grad()
    def get_free_form_answer(self, image, question):
        free_form_answer = self.model.vqa(
            image=image, 
            question=question,
        )
        return free_form_answer
    
    @torch.no_grad()
    def get_answer(self, image, question, choices=None):
        # Get VQA model's answer
        free_form_answer = self.model.vqa(
            image=image, 
            question=question, 
            choices=choices
        )
        
        multiple_choice_answer = free_form_answer.lower()
        if self.sbert:
            # Limit the answer to the choices
            if choices is not None and free_form_answer.lower() not in choices:
                multiple_choice_answer = self.sbert_model.multiple_choice(free_form_answer, choices)
        return {"free_form_answer": free_form_answer, "multiple_choice_answer": multiple_choice_answer}
    
    @torch.no_grad()
    def get_multi_image_answer(self, images, question, choices=None):
        # Get VQA model's answer
        free_form_answer = self.model.multi_image_vqa(
            images=images, 
            question=question, 
            choices=choices
        )
        
        multiple_choice_answer = free_form_answer.lower()
        if self.sbert:
            # Limit the answer to the choices
            if choices is not None and free_form_answer.lower() not in choices:
                multiple_choice_answer = self.sbert_model.multiple_choice(free_form_answer, choices)
        return {"free_form_answer": free_form_answer, "multiple_choice_answer": multiple_choice_answer}

    def multi_question(self, image, questions):
        answers = []
        for question, choices in questions:
            answers.append(self.get_answer(image, question, choices)['multiple_choice_answer'])
        return answers
    
# llava model
class Llava():
    def __init__(
        self,
        device,
        ckpt,
        custom_sys = False,
        model_base = None,
        max_memory = None,
        requires_grad = False
    ):
        # Model
        disable_torch_init()
        self.custom_sys = custom_sys
        if isinstance(device, list):
            max_memory = max_memory if max_memory is not None else utils.get_max_memory(device)
        self.model_name = get_model_name_from_path(ckpt)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(ckpt, model_base, self.model_name, max_memory=max_memory)
        self.model.requires_grad_(requires_grad)
        self.model.get_model().vision_tower = self.model.get_model().vision_tower.to(self.model.device)
        torch.cuda.empty_cache()
          
    def process_image(self, image, squared=False):
        return self.image_processor.preprocess(image, return_tensors='pt', squared=squared)['pixel_values'].half().cuda()

    def get_model(self):
        return self.model

    def multi_image_vqa(self, **kwargs):
        if 'choices' in kwargs and kwargs['choices'] is not None:
            qs = f'Question: {kwargs["question"]} Choices: {", ".join(kwargs["choices"])}.'
        else:
            qs = f'Question: {kwargs["question"]}'
            
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            if self.custom_sys:
                conv_mode = "llava_v1_gradbias"
            else:
                conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = kwargs['images']

        if image_tensor.shape[0] > 1:
            prompt = ''.join([prompt.split('\n')[0]]+['<image>']*(image_tensor.shape[0]-1))+'\n'+prompt.split('\n')[1]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=256,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        # print(len(output_ids[0]))
        # quit()

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs

    def vqa(self, **kwargs):
        if 'choices' in kwargs and kwargs['choices'] is not None:
            qs = f'Question: {kwargs["question"]} Choices: {", ".join(kwargs["choices"])}. Answer:'
        else:
            qs = f'Question: {kwargs["question"]} Answer:'
            
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            if self.custom_sys:
                conv_mode = "llava_v1_gradbias"
            else:
                conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = kwargs['image']

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        input_ids = input_ids.to(self.model.device)
        
        # with torch.inference_mode():
        output_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=256,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

        # print(len(output_ids[0]))
        # quit()

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs

class BLIP:
    def __init__(self, device, ckpt="Salesforce/blip-vqa-capfilt-large", custom_sys=False):
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForQuestionAnswering.from_pretrained(ckpt)
        self.device = device
        print(self.model)
        quit()
        self.model.to(self.device)

    def vqa(self, **kwargs):
        image = kwargs['image'].convert('RGB')
        question = kwargs['question']
        # prepare image + question
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_length=50)
        generated_answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
       
        return generated_answer[0]

    def multi_image_vqa(self, **kwargs):
        # throw not implemented error
        raise NotImplementedError("Multi image VQA not implemented for BLIP")

class BLIP2:
    def __init__(self, device, ckpt='pretrain_flant5xl', custom_sys=False):
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type=ckpt, is_eval=True, device=self.device)
        
    def process_image(self, image):
        image = image.convert('RGB')
        return self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

    def vqa(self, **kwargs):
        image = kwargs['image']
        question = kwargs['question']
        choices = kwargs['choices']
        
        if len(choices) == 0:
            answer = self.model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
        else:
            answer = self.model.generate({"image": image, "prompt": f"Question: {question} Choices: {', '.join(choices)}. Answer:"})
        return answer[0]

    def multi_image_vqa(self, **kwargs):
        # throw not implemented error
        raise NotImplementedError("Multi image VQA not implemented for BLIP2")

class Clip_model():
    def __init__(self, device, ckpt, requires_grad=False):
        print("-> Initializing CLIP...")
        self.device = device
        self.clip_model, self.clip_transform = clip.load(ckpt, device=device)
        self.clip_model.to(self.device)
        self.clip_model.float()
        if not requires_grad:
            self.clip_model.eval()
        print("---> CLIP Initialized")
    
    def __call__(self, image, classes):
        logits_per_image, logits_per_text = self.clip_model(image, clip.tokenize(classes).to(self.device))
        return logits_per_image.softmax(dim=-1).cpu()
    
    def get_model(self):
        return self.clip_model
    
    def vqa(self, image, question, choices):
        return self(image, choices)

class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2", device='cuda:0'):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        self.model = self.model.to(device)
        print("Using SBERT on device: ", device)
            
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))
            
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.detach().cpu()

    def similarity(self, sentence1, sentence2):
        sentence1_embedding = self.embed_sentences([sentence1])
        sentence2_embedding = self.embed_sentences([sentence2])
        return torch.matmul(sentence1_embedding, sentence2_embedding.T).item()

    def get_embedding(self, sentence):
        sentence_embedding = self.embed_sentences([sentence])
        return sentence_embedding
    
    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        choices_embedding = self.embed_sentences(choices)
        top_choice_index = torch.argmax(torch.matmul(choices_embedding, answer_embedding.T)).item()
        return choices[top_choice_index]