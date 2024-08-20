from llama2.generation import Llama as LLama2
from llama3.generation import Llama as LLama3
import os

class Llama_2:
    def __init__(
        self,
        rank,
        opt
    ) -> None:
        # set RANK environment variable
        self.generator = LLama2.build(
            ckpt_dir=opt['LLM']['weights_path'],
            tokenizer_path=opt['LLM']['tokenizer_path'],
            max_seq_len=opt['LLM_config']['max_seq_len'],
            max_batch_size=opt['LLM_config']['batch_size'],
            model_parallel_size=opt['LLM']['model_parallel_size'],
            local_rank=rank,
            seed=opt['seed'],
        )
        self.force_answer_prompt = opt['LLM']['force_answer_prompt']
        self.max_gen_len = opt['LLM_config']['max_gen_len']
        self.temperature = opt['LLM_config']['temperature']
        self.top_p = opt['LLM_config']['top_p']
        self.SYSTEM_PROMPT = opt['LLM_config']['SYSTEM_PROMPT']
        self.outputs = []
        self.generator.model.output.register_forward_hook(self.save_output)
    
    def save_output(self, module, input, output):
        self.outputs.append(output)

    def get_output(self):
        return self.outputs

    def clear_output(self):
        self.outputs = []

    def generate(self, sentence):
        dialogs = []
        sentence = sentence.replace("\n","")
        dialogs.append(
            self.SYSTEM_PROMPT + [
                {
                    'role': 'user',
                    'content': sentence
                }
            ]
        )
        return self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            force_answer_prompt=self.force_answer_prompt
        )

    def update_system_prompt(self, system_prompt):
        self.SYSTEM_PROMPT = system_prompt

class Llama_3:
    def __init__(
        self,
        rank,
        opt
    ) -> None:
        # set RANK environment variable
        self.generator = LLama3.build(
            ckpt_dir=opt['LLM']['weights_path'],
            tokenizer_path=opt['LLM']['tokenizer_path'],
            max_seq_len=opt['LLM_config']['max_seq_len'],
            max_batch_size=opt['LLM_config']['batch_size'],
            model_parallel_size=opt['LLM']['model_parallel_size'],
            seed=opt['seed'],
        )
        self.force_answer_prompt = opt['LLM']['force_answer_prompt']
        self.max_gen_len = opt['LLM_config']['max_gen_len']
        self.temperature = opt['LLM_config']['temperature']
        self.top_p = opt['LLM_config']['top_p']
        self.SYSTEM_PROMPT = opt['LLM_config']['SYSTEM_PROMPT']
        self.outputs = []
        self.generator.model.output.register_forward_hook(self.save_output)
    
    def save_output(self, module, input, output):
        self.outputs.append(output)

    def get_output(self):
        return self.outputs

    def clear_output(self):
        self.outputs = []

    def generate(self, sentence):
        dialogs = []
        sentence = sentence.replace("\n","")
        dialogs.append(
            self.SYSTEM_PROMPT + [
                {
                    'role': 'user',
                    'content': sentence
                }
            ]
        )
        return self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            # force_answer_prompt=self.force_answer_prompt
        )

    def update_system_prompt(self, system_prompt):
        self.SYSTEM_PROMPT = system_prompt