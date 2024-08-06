'''
    Simplfine is an easy-to-use, open-source library for fine-tuning LLMs models quickly on your own hardware or cloud.
    Copyright (C) 2024  Simplifine Corp.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    train_type:str
    max_length:int
    num_return_sequences:int
    do_sample:bool
    top_k:int
    top_p:float
    temperature:float
    prompt_template:str
    response_template:str
    keys:list


def parse_sft_prompt(generate_config:GenerationConfig, data:dict, tokenizer:AutoTokenizer):
    _is_chat = False
    if tokenizer.chat_template:
        # dummy messages to extract chat tokens
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Who won the world series in 2020?"
            },
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020."
            }
    ]   
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        chat_temp = text.split(messages[1]['content'])[-1].split(messages[-1]['content'])[0]
        chat_response_temp = chat_temp.replace(tokenizer.eos_token,'')
        _is_chat = True
    else:
        _is_chat = False
    formatted_prompts = []
    expected_responses = []
    for i in range(len(data[generate_config.keys[0]])):
        formatted_text = generate_config.prompt_template.format(
                        **{key: data[key][i] for key in generate_config.keys}
                    )
        if _is_chat:
            user_text, assistant_text = formatted_text.split(generate_config.response_template)
            assistant_text = generate_config.response_template + assistant_text
            messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            chat_prompt_pre_gen, chat_prompt_post_gen = chat_prompt.split(chat_response_temp)
            formatted_prompts.append(chat_prompt_pre_gen)
            expected_responses.append(chat_prompt_post_gen)
        else:
            formatted_text_pre_gen, formatted_text_post_gen = formatted_text.split(generate_config.response_template)
            formatted_prompts.append(formatted_text_pre_gen)
            expected_responses.append(formatted_text_post_gen)
    return formatted_prompts, expected_responses
    


def generate_from_pretrained(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, generate_confg:GenerationConfig,
                             data:dict={}):
    generated_text = []
    if generate_confg.train_type == 'sft':
        formatted_prompt, expected_outputs = parse_sft_prompt(generate_confg, data, tokenizer)
        for prompt, expected_output in zip(formatted_prompt, expected_outputs):
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=generate_confg.max_length, num_return_sequences=generate_confg.num_return_sequences,
                                    do_sample=generate_confg.do_sample, top_k=generate_confg.top_k, top_p=generate_confg.top_p, temperature=generate_confg.temperature)
            text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_text.append(text)
    return generated_text
