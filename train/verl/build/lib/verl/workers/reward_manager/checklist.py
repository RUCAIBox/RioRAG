import numpy as np
import torch
from verl.utils.reward_score import _default_compute_score
from verl import DataProto
from verl.workers.reward_manager.util.reward_api_batch import (
    process_prompts
)

penalty_length = 800

class ChecklistRewardManager:
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
            # print("****non_tensor_batch****", data_item.non_tensor_batch)

            # checklist score
            # 需要传入多个items,包含这些字段: query prompt guidelines aspects response
            item = {
                "question": data_item.non_tensor_batch['question'],
                "checklist": data_item.non_tensor_batch['checklist'],
                "response": response
            }
            
            checklist_items.append(item)

            # reward_tensor[i, valid_response_length - 1] = score
            # 记录赋值位置
            valid_response_lengths.append(valid_response_length)
        
        print("getting checklist scores...")
        checklist_scores = process_prompts(checklist_items, self.tokenizer)
        print("get checklist scores done!!!")

        for i in range(len(data)):
            # 如果不带cot长度超过penalty_length则衰减
            if without_cot_response_lengths[i] > penalty_length:
                k, m = 5, 2
                checklist_score = checklist_scores[i] * np.exp(-k * ((without_cot_response_lengths[i] - penalty_length) / 2048) ** m)
            else:
                checklist_score = checklist_scores[i]
            # RM api clip
            reward_tensor[i, valid_response_lengths[i] - 1] = checklist_score

        return reward_tensor