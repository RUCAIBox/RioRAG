import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import time
import openai
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import demjson

api_tokenizer = AutoTokenizer.from_pretrained(trunc_tokenizer_path)
def repair_json(json_str):
    lines = json_str.split('\n')
    repaired_lines = []

    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            value = value.split('//', 1)[0]
            value = value.strip()
            if value.endswith(','):
                value = value[:-1].strip()
            if value.startswith('"') or value.startswith("'"):
                value = value[1:].strip()
            if value.endswith('"') or value.endswith("'"):
                value = value[:-1].strip()
            fixed_value = ""
            for i, char in enumerate(value):
                if char == '"' and (i == 0 or value[i - 1] != '\\'):
                    char = "\\\""
                elif char == '"' and len(fixed_value) >= 2 and fixed_value[-2:] == "\\\\":
                    fixed_value = fixed_value[:-1]
                fixed_value += char
            repaired_line = f'{key}: "{fixed_value}",'
            repaired_lines.append(repaired_line)
        else:
            repaired_lines.append(line)

    return '\n'.join(repaired_lines)


def extract_reward(output):
    output = output.split("</think>")[-1].strip()
    if output.startswith("```"):
        output = output[len("```"): ].strip()
    if output.startswith("json"):
        output = output[len("json"): ].strip()
    if output.endswith("```"):
        output = output[:-len("```")].strip()
    output = repair_json(output)
    try:
        results = demjson.decode(output)
        if not isinstance(results, list):
            raise Exception("Invalid JSON format")
    except Exception as e:
        raise e
    return results

def get_verification_prompt_simple(tokenizer, query: str, response: str, checklist: list) -> str:
    prompt = f"""
Check each fact in the checklist against the response.

For each fact:
- If it matches the response, mark as "Consistent".
- If it conflicts with the response, mark as "Contradictory".
- If it is not mentioned, mark as "Missing".

Output format:
```
[
  {{
    "point": copy the original factual point exactly as given,
    "analysis": one short sentence judging the relation between the response and the fact, clearly indicating if it is Consistent, Contradictory, or Missing,
    "conclusion": Consistent/Contradictory/Missing,
  }},
  ...
]
```

Inputs:

Question: {query}

Response: 
{response}

Checklist:
{checklist}

Strictly follow the required output format.
    """
    if tokenizer is not None:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True)
    else:
        return [{'role': 'user', 'content': prompt.strip()}]
    
def process_data(tokenizer, input_data, gen_model):
    prompts = []
    for sample in tqdm(input_data, total=len(input_data), desc="Processing"):
        query = sample['item']['Question']
        qwq_response = sample[f'{gen_model}_gen'].split("</think>")[-1].split("**Final Information**")[-1].strip()
        checklist = sample['checklist']
        checklist = '\n'.join([f"{pid + 1}. {p}" for pid, p in enumerate(checklist)])
        prompts.append(get_verification_prompt_simple(tokenizer, query, qwq_response, sample['checklist']))
    return prompts

def extract_summary(output):
    return output.split("</think>")[-1].split("**Final Information**")[-1].strip()
