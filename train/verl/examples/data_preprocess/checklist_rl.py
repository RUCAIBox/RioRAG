import os
import argparse
from datasets import Dataset
from tqdm import tqdm
import json
from shutil import copytree as copy
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/root/paddlejob/workspace/env_run/DeepSeek-R1-Distill-Qwen-14B")

def get_all_webs_qa_prompt(tokenizer, query: str, web_contents: list) -> str:
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
    return [{'role': 'user', 'content': prompt.strip()}]

length = []
def process_sample(sample, index):
    item = sample["item"]
    question = item["Question"]
    webs = []
    for iid, info in enumerate(sample['all_info']):
        if info['type'] == 'raw':
            webs.extend(info['content'])
    
    return {
        "id": f"webglm_{index}",
        "checklist": sample["checklist"],
        "question": question,
        "webs": webs,
    }

def process_sample_2(sample, index):
    max_len = 6000
    webs = sample.pop("webs")[:5]
    new_webs = []
    for w in webs:
        keys = list(w.keys())
        for k in keys:
            if isinstance(w[k], str) and (not w[k] or "http error" in w[k].lower()):
                w.pop(k)
        else:
            content = tokenizer(w[k], add_special_tokens=False)["input_ids"]
            max_len -= len(content)
            if max_len < 0:
                w[k] = tokenizer.decode(content[:max_len])
            new_webs.append(w)
            if max_len <= 0:
                break
    question = sample["question"]
    webs = {f'Web {wid + 1}': w for wid, w in enumerate(new_webs)}
    webs = json.dumps(webs, indent=4)

    sample["prompt"] = get_all_webs_qa_prompt(None, question, webs)
    # length.append(len(tokenizer.apply_chat_template(prompt_message, tokenizer=tokenizer, add_generation_prompt=False)))

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="/root/paddlejob/workspace/env_run/search_o1/output/eli5/QwQ-32B_gen.json")
    parser.add_argument('--hdfs_dir', type=str, default=None)
    args = parser.parse_args()

    local_dir = "/root/paddlejob/workspace/env_run/data/training/"
    os.makedirs(local_dir, exist_ok=True)

    print("Loading raw JSON data...")
    with open(args.input_path) as f:
        data = json.load(f)

    new_data = []
    for sid, sample in tqdm(enumerate(data), total=len(data)):
        new_data.append(process_sample(sample, sid))

    dataset = Dataset.from_list(new_data)

    print("Processing with multiprocessing map...")
    dataset = dataset.map(
        process_sample_2,
        with_indices=True,
        num_proc=1,
    )
    import pdb
    pdb.set_trace()

    print("Saving to parquet...")
    dataset.to_parquet(os.path.join(local_dir, 'rl.train.parquet'))

    if args.hdfs_dir is not None:
        os.makedirs(args.hdfs_dir, exist_ok=True)
        copy(local_dir, args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")

    print("✅ Done.")

if __name__ == "__main__":
    main()