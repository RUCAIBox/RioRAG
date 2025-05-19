import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import random
import re
import time
import demjson
import openai
from tqdm import tqdm
    

def parse_args():
    parser = argparse.ArgumentParser(description="Run.")

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="Path to the pre-trained model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=None,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--min_p',
        type=float,
        default=None,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=None,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--eval_path',
        type=str,
        default=None,
        help="data_path"
    )

    parser.add_argument(
        '--source_path',
        type=str,
        default=None,
        help="data_path"
    )

    parser.add_argument(
        '--output_dir_base',
        type=str,
        default=None,
        help="output_dir"
    )

    return parser.parse_args()


def get_verification_prompt_simple(tokenizer, query: str, response: str, checklist: list) -> str:
    prompt = f"""
Check each fact in the checklist against the response.

For each fact:
- If it matches the response, mark as "Consistent".
- If it conflicts with the response, mark as "Contradictory".
- If it is not mentioned, mark as "Missing".
- If it is partially mentioned, mark as "Partial".

Output format:
```
[
  {{
    "point": copy the original factual point exactly as given,
    "analysis": one short sentence judging the relation between the response and the fact, clearly indicating if it is Consistent, Contradictory, or Missing,
    "conclusion": Consistent/Contradictory/Missing/Partiall,
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

Strictly follow the required json output format.
    """
    if tokenizer is not None:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True)
    else:
        return [{'role': 'user', 'content': prompt.strip()}]
    
def process_data(tokenizer, source_data, eval_data):
    prompts = []
    for sample, gen in tqdm(zip(source_data, eval_data), total=len(source_data), desc="Processing"):
        query = sample['Question']
        qwq_response = gen["generation"].split("</think>")[-1].split("**Final Information**")[-1].strip()
        checklist = sample['checklist']
        checklist = '\n'.join([f"{pid + 1}. {p}" for pid, p in enumerate(checklist)])
        prompts.append(get_verification_prompt_simple(None, query, qwq_response, checklist))
    return prompts

client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE"),
        )
model = None



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


def get_response_for_prompt(prompt, urls=[]):
    """
    对单个 prompt 无限循环尝试多个 URL，直到成功或手动中断。
    :param prompt: 输入的 prompt
    :param urls: 可用的 URL 列表
    :return: 成功时返回响应内容
    """
    while True:
        try:
            messages = prompt if isinstance(prompt, list) else [
                    {"role": "user", "content": prompt}
                ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=3000,
                top_p=0.1,
            ).choices[0].message.content
            response = extract_reward(response)
            return response
        except Exception as e:
            print(f"Error OPENAI: {e}")
            if "Content Exists Risk" in str(e):
                return []
            if not "rate limit" in str(e).lower():
                return []
            time.sleep(0.1)
            continue 
    return []

def process_prompts(
        prompts, 
        max_workers=64,
    ):
    """
    使用多线程并发处理多个 prompts，并显示进度条。
    :param prompts: list of prompts to process
    :param urls: list of available URLs to try for each prompt
    :param max_workers: 最大线程数，默认 32
    :return: list of (prompt, response) tuples
    """
    results = [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
         tqdm(total=len(prompts), desc="Processing Prompts", unit="prompt") as pbar:
        futures = {executor.submit(get_response_for_prompt, prompt): idx for idx, prompt in enumerate(prompts)}

        # 按完成顺序收集结果
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                print(f"Error processing prompt '{prompts[idx][:20]}...': {e}")
                results[idx] = None 
            finally:
                pbar.update(1)

    return results

def main():
    args = parse_args()
    global model
    directory = os.path.dirname(args.eval_path) if args.output_dir_base is None else args.output_dir_base
    new_file_path = os.path.join(directory, f"reward.json")
    no_reward = None
    source_data = json.load(open(args.source_path, 'r', encoding='utf-8'))
    eval_data = json.load(open(args.eval_path, 'r', encoding='utf-8'))



    model = model_nick = args.model_path
    prompts = process_data(None, source_data, eval_data)
    if no_reward:
        new_prompts = [p for pid, p in enumerate(prompts) if pid in no_reward]
    else:
        new_prompts = prompts
    prompts = new_prompts
    print(f"For generation: {len(new_prompts)}")
    output = process_prompts(prompts)
    raw_outputs = output
    if no_reward:
        for eid, res in zip(no_reward, raw_outputs):
            eval_data[eid][f'reward'] = res
    else:
        for gen, res in zip(eval_data, raw_outputs):
            gen[f'reward'] = res
    print(f"Save results to {new_file_path}.")
    json.dump(eval_data, open(new_file_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)



if __name__ == '__main__':
    main()
