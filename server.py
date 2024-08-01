# -*- coding: utf-8 -*-
import json
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from fastapi import FastAPI
from pydantic import BaseModel

from component.template import template_dict

app = FastAPI()

device = 'cuda'

model_name_or_path = 'finetuned/firefly-qwen1.5-7b-sft-qlora-merge'
# model_name_or_path = 'models/Qwen1.5-7B-Chat'


class ChatCompletion(BaseModel):
    model: str
    system_prompt: str = ""
    messages: list = []
    temperature: float = 0.0


@app.post("/completion/")
def completion(chat_completion: ChatCompletion):
    logger.info(f"completion: {chat_completion}")
    template = template_dict.get(chat_completion.model)
    system_prompt = chat_completion.system_prompt if chat_completion.system_prompt else template.system
    messages = chat_completion.messages

    # chatglm使用官方的数据组织格式
    if model.config.model_type == 'chatglm':
        text = '[Round 1]\n\n问：{}\n\n答：'.format(system_prompt)
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    # 为了兼容qwen-7b，因为其对eos_token进行tokenize，无法得到对应的eos_token_id
    else:
        input_ids = []

        if template.system_format is not None:
            if system_prompt is not None:
                system_text = template.system_format.format(content=system_prompt)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in messages:
            role, content = item['role'], item['content']
            if role == 'user':
                content = template.user_format.format(content=content)
            else:
                content = template.assistant_format.format(content=content)
            tokens = tokenizer.encode(content, add_special_tokens=False)
            input_ids += tokens
        input_ids = torch.tensor([input_ids], dtype=torch.long)

    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    # response = tokenizer.batch_decode(outputs)
    response = tokenizer.decode(outputs)
    response = response.strip().replace(tokenizer.eos_token, "").strip()

    result = {
        'output': response
    }
    return result



logger.info(f"Starting to load the model {model_name_or_path} into memory")

# 加载model和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)
# QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token>为<|endoftext|>
# if tokenizer.__class__.__name__ == 'QWenTokenizer':
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     tokenizer.bos_token_id = tokenizer.eos_token_id
#     tokenizer.eos_token_id = tokenizer.eos_token_id

logger.info(f"Successfully loaded the model {model_name_or_path} into memory")

# 计算模型参数量
total = sum(p.numel() for p in model.parameters())
print("Total model params: %.2fM" % (total / 1e6))
model.eval()
