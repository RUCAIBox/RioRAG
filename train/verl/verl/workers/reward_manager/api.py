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
import torch

# RM api
import requests
import json
import concurrent.futures
import math
import sys

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
    try:
        return sigmoid(result_json["score"][0])
    except Exception as e:
        return 0
    

def get_RM_scores(data):
    query = data["src"][0]
    K = ""
    if data["src"][-1].find("[<prompt-res>]")==0:
        K = data["src"][-1].replace("[<prompt-res>]", "").replace("[</prompt-res>]", "")
    if data["src"][-1].find("[<search-res>]")==0:
        K = data["src"][-1].replace("[</search-res>]", "").replace("[<search-res>]", "") + "\n根据以上参考文章回答问题，补全对话"
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_RM_score, query, response, K) for response in data["responses"][:10]]
        RM_scores = [future.result() for future in futures]
        data["rm_scores"] = RM_scores
        print(RM_scores)
        return data


class OpenAPIRewardManager:
    """The api-based reward manager.
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
        
        # data.batch keys:
        # 1. responses: response tokens
        # 2. prompts: 

        already_print_data_sources = {}

        # 调用api
        # print("reward_tensor.shape: ", reward_tensor.shape)  # batch_size=64 * max_response_length=2048

        checklist_items = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]   # 可能会截断

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]  # 同理截断

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # 这里打印一些东西
            # print('prompts:..........\n')
            # print(valid_prompt_ids)  # 1D tensor
            # print("\n截断前的prompt")
            # print(self.tokenizer.decode(prompt_ids))
            # print('responses:..........\n')
            # print(valid_response_ids)  # 1D tensor
            # print("\n截断前的response")
            # print(self.tokenizer.decode(response_ids))
            # print("解码后的有效prompt:\n")
            # print(self.tokenizer.decode(valid_prompt_ids))
            # print("\n\n")
            # print("解码后的有效response:\n")
            # print(self.tokenizer.decode(valid_response_ids))

            # data_source = data_item.non_tensor_batch['data_source']  # error
            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']  # error

            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=sequences_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )

            # RM api score
            prompt = self.tokenizer.decode(valid_prompt_ids)
            response = self.tokenizer.decode(valid_response_ids)
            score = get_RM_score(query=prompt, response=response, knowledge="")
            

            reward_tensor[i, valid_response_length - 1] = score

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print(sequences_str)

        return reward_tensor
