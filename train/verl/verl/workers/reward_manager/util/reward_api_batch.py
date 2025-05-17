import requests
from .reward_api import get_verification_prompt_simple, extract_reward, extract_summary, api_tokenizer

API_URL = f"http://{Reward_Url}/generate"

retry_schemes = [
    (0.1, 0.1),
    (0.3, 0.3),
    (0.6, 0.95),
    (0.9, 1.0)
]

def call_server(prompts, temperature, top_p, top_k, max_tokens=10000):
    data = {
        "prompts": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    try:
        response = requests.post(API_URL, json=data, timeout=3000)
        result = response.json()["responses"]
        return result
    except Exception as e:
        result = [""] * len(prompts)
        return result

def process_prompts(all_data, tokenizer):
    prompts = []
    scores = [0.5] * len(all_data)
    sids = list(range(len(all_data)))
    prompts = []
    print("Using api tokenizer")
    for sample in all_data:
        question, checklist, response = sample['question'], sample['checklist'], sample['response']
        response = extract_summary(response)
        checklist_text = '\n'.join([f"{i+1}. {p}" for i, p in enumerate(checklist)])
        prompt = get_verification_prompt_simple(api_tokenizer, question, response, checklist_text)
        sample['prompt'] = prompt
        prompts.append(prompt)
    for temp, top_p in retry_schemes:
        if not prompts:
            break
        output = call_server(prompts, temperature=temp, top_p=top_p, top_k=1.0)
        prompts = []
        new_sids = []
        for sid, response in zip(sids, output):
            sample = all_data[sid]
            checklist = sample['checklist']
            try:
                reward = extract_reward(response)
                assert len(reward) == len(checklist)
                score = 0
                for r in reward:
                    if r["conclusion"].lower().strip() == "consistent":
                        score += 1
                    elif r["conclusion"].lower().strip() != "contradictory":
                        score += 0.6
                scores[sid] = score / len(reward)
            except Exception as e:
                prompts.append(sample['prompt'])
                new_sids.append(sid)
                continue
        sids = new_sids
    
    
    return scores