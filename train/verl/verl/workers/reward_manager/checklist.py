import numpy as np
import torch
from verl.utils.reward_score import _default_compute_score
from verl import DataProto
from verl.workers.reward_manager.util.reward_api_batch import (
    process_prompts
)

penalty_length = 800

def extract_summary(output):
    return output.split("</think>")[-1].split("**Final Information**")[-1].strip()

class ChecklistRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        checklist_items = []
        valid_response_lengths = [] 

        prompts, responses = [], []
        without_cot_response_lengths = [] 
        with_cot_response_lengths = [] 
        penalty_flags = [] 

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]   # left pad

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if "</think>" not in response:
                response = "</think>\n" + response

            with_cot_response_lengths.append(len(response))
            response = extract_summary(response)
            without_cot_response_lengths.append(len(response))

            prompts.append(prompt)
            responses.append(response)
            item = {
                "question": data_item.non_tensor_batch['question'],
                "checklist": data_item.non_tensor_batch['checklist'],
                "response": response
            }
            
            checklist_items.append(item)
            valid_response_lengths.append(valid_response_length)
        
        print("getting checklist scores...")
        checklist_scores = process_prompts(checklist_items, self.tokenizer)
        print("get checklist scores done!!!")

        for i in range(len(data)):
            if without_cot_response_lengths[i] > penalty_length:
                k, m = 5, 2
                checklist_score = checklist_scores[i] * np.exp(-k * ((without_cot_response_lengths[i] - penalty_length) / 2048) ** m)
            else:
                checklist_score = checklist_scores[i]
            reward_tensor[i, valid_response_lengths[i] - 1] = checklist_score

        return reward_tensor