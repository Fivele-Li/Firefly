from loguru import logger

from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

from component.template import template_dict

app = FastAPI()

# model_name_or_path = 'finetuned/firefly-qwen1.5-7b-sft-qlora-merge'
model_name_or_path = 'models/Qwen1.5-7B-Chat'

# Create an LLM.
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.5)


class ChatCompletion(BaseModel):
    model: str
    system_prompt: str = ""
    messages: list = []
    temperature: float = 0.0
    top_p: float = 0.8
    presence_penalty: float = 0.8
    frequency_penalty: float = 0.8
    max_tokens: int = 128
    

@app.post("/completion/")
def completion(chat_completion: ChatCompletion):
    logger.info(f"completion: {chat_completion}")
    template = template_dict.get(chat_completion.model)
    system_prompt = chat_completion.system_prompt if chat_completion.system_prompt else template.system
    messages = chat_completion.messages
    
    prompt = ""
    if template.system_format is not None:
        if system_prompt is not None:
            prompt = template.system_format.format(content=system_prompt)
    # concat conversation
    for item in messages:
        role, content = item['role'], item['content']
        if role == 'user':
            content = template.user_format.format(content=content)
        else:
            content = template.assistant_format.format(content=content)
        
        prompt += content
    
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=chat_completion.temperature, top_p=chat_completion.top_p, 
                                     presence_penalty=chat_completion.presence_penalty, frequency_penalty=chat_completion.frequency_penalty,
                                     max_tokens=chat_completion.max_tokens)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompt, sampling_params)
    
    response = ""
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        response += generated_text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    return {
        'output': response
    }