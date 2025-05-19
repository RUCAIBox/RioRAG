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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager.util.checklist_util import get_checklist_scores
import torch
import numpy as np

# RM api
import requests
import json
import concurrent.futures
import math
import sys

# 惩罚长度
penalty_length = 800


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def get_RM_score(query, response, knowledge=""):
    RM_API = "http://10.174.138.86:8720"
    #try:
    data = {
        "context": [
            {"role": "user", "utterance": query}
        ],
        "responses": [
            response
        ],
        "knowledge": knowledge
    }
    result_json = requests.post(
        "%s/api/infer" % RM_API,
        json=data
    ).json()
    #print(result_json["score"][0])
    # return sigmoid(result_json["score"][0])
    try:
        return sigmoid(result_json["score"][0])
    except Exception as e:
        return 0


def eb_tokenzie(sentences):
    import requests
    API = "http://10.93.65.107:8326/tokenize"
    #按照sentences个数组batch, 不能过多
    data = {"sentences": sentences}
    ret = requests.post(API, json=data).json()
    
    return ret["sentence_tokens"]


def get_RM_scores(prompts, responses):
    # 保证长度之和不超过8K
    sentences = [prompt + response for prompt, response in zip(prompts, responses)]
    ids = eb_tokenzie(sentences)
    lens = [len(item) for item in ids]

    queries, Ks = [], []
    for idx, prompt in enumerate(prompts):
        K, query = prompt.split("\n对话问题：")
        Ks.append(K)
        queries.append(query)
        # check length
        if lens[idx] > 8000:
            truncation_len = lens[idx] - 8000
            responses[idx] = responses[idx][:-truncation_len]
    
    # 40并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        futures = [executor.submit(get_RM_score, query, response, K) for query, response, K in zip(queries, responses, Ks)]
        RM_scores = [future.result() for future in futures]
    
    return RM_scores


class CustomRewardManager:
    """The custom reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        checklist_items = []
        valid_response_lengths = []  # 这个是总的长度，包含思维链，但是是decode前的长度

        prompts, responses = [], []
        without_cot_response_lengths = []  # 统计一下去掉思维链后的长度，decode后
        with_cot_response_lengths = []  # 总的回复长度，decode后
        penalty_flags = []  # 是否包含<think>和</think>，以及正常结束

        for i in range(len(data)):
            flag = False  # 是否惩罚
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]   # left pad

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 如果没有正常结束就惩罚
            if valid_response_ids[-1] != self.tokenizer.eos_token_id:
                flag = True

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)  # skip_special_tokens=True去掉末尾的end text

            prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if "<think>" not in response:
                response = "<think>\n" + response

            # 如果不包含则惩罚
            if response.find("<think>") == -1 or response.find("</think>") == -1 or response.find("</think>") < response.find("<think>"):
                flag = True
            
            penalty_flags.append(flag)

            if "</think>" not in response:
                # 保证后续切割的时候能给正确切割，把所有输出都当做去掉思维链后的response
                response = "</think>\n" + response

            # 记录不去掉思维链的长度
            with_cot_response_lengths.append(len(response))
            # 去掉思维链
            response = response.split("</think>")[-1].strip()
            without_cot_response_lengths.append(len(response))

            prompts.append(prompt)
            responses.append(response)

            # checklist score
            # 需要传入多个items,包含这些字段: query prompt guidelines aspects response
            item = {
                "query": data_item.non_tensor_batch['query'],
                "prompt": prompt,
                "guidelines": data_item.non_tensor_batch['guidelines'],
                "aspects": data_item.non_tensor_batch['aspects'],
                "response": response
            }
            
            checklist_items.append(item)

            # reward_tensor[i, valid_response_length - 1] = score
            # 记录赋值位置
            valid_response_lengths.append(valid_response_length)
        
        print("getting checklist scores...")
        checklist_scores = get_checklist_scores(checklist_items)
        print("get checklist scores done!!!")
        # 计算RM score 80并发
        print("getting RM api scores...")
        rm_scores = get_RM_scores(prompts, responses)
        print("get RM api scores done!!!")

        for i in range(len(data)):
            # 如果不带cot长度超过penalty_length则衰减
            if without_cot_response_lengths[i] > penalty_length:
                k, m = 5, 2
                checklist_score = checklist_scores[i] * np.exp(-k * ((without_cot_response_lengths[i] - penalty_length) / 2048) ** m)
            else:
                checklist_score = checklist_scores[i]
            # RM api clip
            rm_score = min(rm_scores[i], 0.8)
            # 计算penalty
            penalty = 1.0 if penalty_flags[i] else 0
            # ensemble
            if rm_score < 0.01:
                reward_tensor[i, valid_response_lengths[i] - 1] = 1.4 * rm_score
            else:
                reward_tensor[i, valid_response_lengths[i] - 1] =  checklist_score + 0.4 * rm_score - penalty

        return reward_tensor
