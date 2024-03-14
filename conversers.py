
import common
from language_models import GPT, Claude, PaLM, HuggingFace, GENAI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from config import VICUNA_PATH, LLAMA_PATH, TEMP, TOP_P, PEFT_PATH, QWEN_PATH, CODELLAMA_PATH, DEEPSEEK_PATH

def load_models(args):
    model = LM(model_name = args.model,
                max_n_tokens = args.max_n_tokens,
                temperature = TEMP,
                top_p = TOP_P,
                lora = args.lora
                )
    return model

class LM():
    '''
        Base class for language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    '''
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float,
                lora: bool):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name, lora)
        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_response(self, prompts_list):
        """
        Generates responses for prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - prompts_list: List of prompts.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            elif "gemini" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        outputs_list = self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                    )

        return outputs_list


def load_indiv_model(model_name, lora=False, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    elif model_name in ["gemini-pro"]:
        lm = GENAI(model_name)
    else:
        if lora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,   # load the model into memory using 4-bit precision
                bnb_4bit_use_double_quant=True, # use double quantition
                bnb_4bit_quant_type="nf4", # use NormalFloat quantition
                bnb_4bit_compute_dtype=torch.bfloat16 # use hf for computing when we need
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                use_cache=True,
                device_map='auto',
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(
                model,
                PEFT_PATH,
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False
            ) 
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            lm = HuggingFace(model_name, model, tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    trust_remote_code=True).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False
            ) 

            if 'llama' in model_path.lower():
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.padding_side = 'left'
            if 'vicuna' in model_path.lower():
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":LLAMA_PATH,
            "template":"llama-2"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        },
        "gemini-pro":{
            "path":"gemini-pro",
            "template":"gemini-pro"
        },
        "deepseek-coder":{
            "path": DEEPSEEK_PATH,
            "template":"deepseek-coder"
        },
        "qwen":{
            "path": QWEN_PATH,
            "template":"qwen"
        },
        "codellama":{
            "path": CODELLAMA_PATH,
            "template":"codellama"
        },
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



    