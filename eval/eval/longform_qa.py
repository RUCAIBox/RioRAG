import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import openai
import torch
from vllm import LLM, SamplingParams


tokenizer = None
tokenizer_path  # truncated tokenizer
trunc_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    


def get_all_webs_qa_prompt(tokenizerr, query: str, web_contents: list) -> str:
    prompt = f"""
**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages**.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to answering the **Current Search Query**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

**Inputs:**
- **Current Search Query:**  
{query}

- **Searched Web Pages:**  
{web_contents}

Now you should analyze each web page and find helpful information based on the current search query "{query}" and previous reasoning steps.
"""
    if tokenizer is not None:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True)
    else:
        return prompt.strip()

def trunc_webs(sample, tokenizer):
    max_len = 5000
    try:
        webs = sample.pop("webs")
        webs = webs[:5]
    except Exception as e:
        webs = [{"context": "Nothing Relevant"}]
    new_webs = []
    for w in webs:
        clean_w = {}
        for k, v in w.items():
            if not isinstance(v, str):
                clean_w[k] = v
                continue
            if not v or "http error" in v.lower():
                continue
            tokens = trunc_tokenizer(v, add_special_tokens=False)["input_ids"]
            if max_len <= 0:
                break
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            text = trunc_tokenizer.decode(tokens)
            clean_w[k] = text
            max_len -= len(tokens)

        if clean_w:
            new_webs.append(clean_w)
        if max_len <= 0:
            break
    question = sample.get("question", sample["Question"])
    webs = {f'Web {wid + 1}': w for wid, w in enumerate(new_webs)}
    webs = json.dumps(webs, indent=4)
    sample["qa_prompt"] = get_all_webs_qa_prompt(tokenizer, question, webs)
    sample["new_webs"] = new_webs
    return sample


def get_checklist_prompt(tokenizerr, query: str, key_points: str) -> str:
    prompt = f"""
You are given a user question and a list of candidate key points.

Your task:
- Keep only the key points that are highly relevant to the question.
- Merge exact duplicates; if two points have slightly different focuses, keep both.
- Each item = one single idea, in one concise sentence.
- If there are conflicting views, use wording like: "Some studies suggest [...], others indicate [...]."
- Output format: \\boxed{{ key point 1 \\n key point 2 \\n key point 3 ... }}
- Only output inside \\boxed{{}}.

Output Format:
\\boxed{{
    Final Key Point 1
    Final Key Point 2
    ...
}}

Input:
- Query: {query}
- Key Points:
{key_points}

- Query: {query}
"""
    if tokenizer is not None:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True)
    else:
        return prompt.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Search O1 for various datasets and models.")

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        default='DeepSeek-R1-Distill-Llama-8B-checklist/global_step_100/actor/huggingface',
        help="Path to the pre-trained model."
    )

    parser.add_argument(
        '--model_nick',
        type=str,
        default='Llama-8B/100',
        help="Path to the pre-trained model."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.1,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=-1,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--min_p',
        type=float,
        default=0,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=0,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=4096,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="output/longfact/webs.json",
        help="data_path"
    )

    parser.add_argument(
        '--output_dir_base',
        type=str,
        default=None,
        help="output_dir"
    )

    return parser.parse_args()


def extract_points(output):
    points = output.split('boxed{')[-1]
    if points.endswith('}'):
        points = points[:-1]
    points = points.strip().split('\n')
    points = [point.strip() for point in points if not "No relevant information" in point]
    return points

def process_data(tokenizer, samples):
    for sid in range(len(samples)):
        sample = samples[sid]
        sample = trunc_webs(sample, tokenizer)
    return [sample["qa_prompt"] for sample in samples]

def process_checklist(tokenizer, input_data):
    prompts = []
    for sample in input_data:
        query = sample['Question']
        points = sample['points_10']
        points = '\n'.join([f"    {p.strip()}" for p in points])
        prompts.append(get_checklist_prompt(tokenizer, query, points))
    return prompts

def load_model(args):
    global toknizer
    llm = LLM(
        model=args.model_path,
        # enforce_eager=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
        dtype=torch.bfloat16,
        # max_model_len=12288,
    )

    print("Model loaded successfully.")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        # top_k=args.top_k,
        # min_p=args.min_p,
        # repetition_penalty=args.repetition_penalty,
    )

    tokenizer = llm.get_tokenizer()

    return llm, tokenizer, sampling_params


def process_all_data(tokenizer, input_data, max_workers=32):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        func = partial(process_data, tokenizer=tokenizer)
        results = list(executor.map(func, input_data))
    return results

def main():
    args = parse_args()
    directory = os.path.dirname(args.data_path) if args.output_dir_base is None else args.output_dir_base
    directory = os.path.join(directory, args.model_nick)
    os.makedirs(directory, exist_ok=True)
    new_file_path = os.path.join(directory, f"new_results.json")
    if os.path.exists(new_file_path):
        print(f"⚠️ 结果文件已存在：{new_file_path}，程序退出以避免覆盖。")
        return


    input_data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    llm, tokenizer, sampling_params = load_model(args)
    qa_prompts = process_data(tokenizer, input_data)
    print(f"For generation: {len(qa_prompts)}")
    qa_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in qa_prompts]
    output = llm.generate(qa_prompts, sampling_params=sampling_params)
    raw_outputs = [out.outputs[0].text for out in output]
    for gen, sample in zip(raw_outputs, input_data):
        gen = gen.split("</think>")[-1].split("**Final Information**")[-1].strip()
        sample["generation"] = gen.split("</think>")[-1].split("**Final Information**")[-1].strip()

    
    json.dump(input_data, open(new_file_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

def get_prompt(tokenizer, query: str, web_content: str) -> str:
    prompt = f"""
You are given a user query and a retrieved web content.

Your task:
- Output several highly relevant key points from the web content.
- Each key point must be one concise sentence and separated by a newline (`\n`).
- The key points must be wrapped inside \\boxed{{
Key Point 1
Key Point 2
...
}}.
- Only use the given web content.
- If no relevant information is found, output \\boxed{{No relevant information}}.
- Do not create or assume content not present in the web content.

Input:
- Query: {query}
- Web Content: 
{web_content}

- Query: {query}
"""
    if tokenizer is not None:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt.strip()}], tokenize=False, add_generation_prompt=True)
    else:
        return prompt.strip()


if __name__ == '__main__':
    main()