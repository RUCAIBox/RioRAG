# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import demjson


def repair_json(json_str):
    # 按 '\n' 分割字符串
    lines = json_str.split('\n')
    repaired_lines = []

    for line in lines:
        # 按 ':' 分割每行
        parts = line.split(':', 1)  # 只分割一次，确保冒号后的内容完整
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
            # 处理第二个字符串中的非法字符
            fixed_value = ""
            for i, char in enumerate(value):
                # 如果字符是 " 且不是 \"，替换成 “
                if char == '"' and (i == 0 or value[i - 1] != '\\'):
                    char = "\\\""
                elif char == '"' and len(fixed_value) >= 2 and fixed_value[-2:] == "\\\\":
                    fixed_value = fixed_value[:-1]
                fixed_value += char
            # 将修复后的值拼接回去
            repaired_line = f'{key}: "{fixed_value}",'
            repaired_lines.append(repaired_line)
        else:
            # 如果分割结果长度不为 2，直接保留原行
            repaired_lines.append(line)

    # 将修复后的行重新组合成字符串
    return '\n'.join(repaired_lines)


def extract_reward(output):
    output = output.split("</think>")
    prefix = "</think>".join(output[: -1]) + "</think>\n\n"
    output = output[-1].strip()
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
        print(e)
        print(output)
        results = []
        # import pdb; pdb.set_trace()
    for item in results:
        item["conclusion"] = "Partial" if item["conclusion"] not in ["Consistent", "Contradictory", "Missing"] else item["conclusion"]
    new_string = json.dumps(results, indent=4)
    new_string = new_string.strip()
    repaired_res = f"{prefix}\n\n```json\n{new_string}\n```"
    return results, repaired_res


# tokenizer = AutoTokenizer.from_pretrained("")
# 加载数据
model_nick = "DeepSeek-R1"
gen_model = "QwQ-32B"

src_len = []
tgt_len = []


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

# 数据预处理
def process_sample(sample, index):
   
    query = sample['item']['Question']
    qwq_response = sample[f"{gen_model}_gen"].split("</think>")[-1].split("**Final Information**")[-1].strip()
    checklist = '\n'.join([f"{i+1}. {p}" for i, p in enumerate(sample['checklist'])])
    prompt = get_verification_prompt_simple(None, query, qwq_response, checklist)
    response = sample[f"{model_nick}_{gen_model}_reward"]
    return {"prompt": prompt, "response": response, "id": f"webglm_{index}"}
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/root/paddlejob/workspace/env_run/search_o1/output/eli5/DeepSeek-R1_QwQ-32B_reward.json')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    if True or not os.path.exists("/root/paddlejob/workspace/env_run/search_o1/output/eli5/DeepSeek-R1_reward_sft.json"):

        with open(args.local_dir) as f:
            data = json.load(f)
        save_data = []
        for sample in data:
            keys = list(sample.keys())
            for key in keys:
                if not key in ["item", "checklist", f"{gen_model}_gen", f"{model_nick}_{gen_model}_reward"]:
                    sample.pop(key)
            reward = f"{model_nick}_{gen_model}_reward"
            results, repaired_res = extract_reward(sample[reward])
            if results and len(results) == len(sample['checklist']):
                sample[reward] = repaired_res
                save_data.append(sample)
        print(f"Saved {len(save_data)} samples to `/root/paddlejob/workspace/env_run/search_o1/output/eli5/DeepSeek-R1_reward_sft.json`")

        with open("/root/paddlejob/workspace/env_run/search_o1/output/eli5/DeepSeek-R1_reward_sft.json", "w") as f:
            json.dump(save_data, f, indent=4)

    dataset = load_dataset("json", data_files="/root/paddlejob/workspace/env_run/search_o1/output/eli5/DeepSeek-R1_reward_sft.json")

    split_dataset = dataset["train"].train_test_split(test_size=500, shuffle=True, seed=42)

    # 提取训练集和测试集
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    train_dataset = train_dataset.map(process_sample, remove_columns=["item", "checklist"], with_indices=True, num_proc=20)
    test_dataset = test_dataset.map(process_sample, remove_columns=["item", "checklist"], with_indices=True, num_proc=20)


    

    local_dir ="/root/paddlejob/workspace/env_run/data/training/"
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
