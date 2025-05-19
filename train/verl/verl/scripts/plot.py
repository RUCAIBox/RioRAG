import re
import json
import matplotlib.pyplot as plt

def parse_log_line(line):
    """解析单行日志生成字典"""
    step_match = re.search(r'step:(\d+)', line)
    if not step_match:
        return None
    
    # 增强正则匹配：支持科学计数法、正负号、无前导零小数
    pattern = r'([a-zA-Z0-9_\/]+):([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    metrics = {}
    
    for key, value in re.findall(pattern, line):
        clean_key = key.replace('/', '_')
        try:
            # 自动识别数值类型（int/float）
            metrics[clean_key] = float(value) if '.' in value or 'e' in value.lower() else int(value)
        except ValueError:
            metrics[clean_key] = value
    
    return {
        "step": int(step_match.group(1)),
        **metrics  # 展开所有指标到顶层
    }


def generate_jsonl(input_file, output_file):
    """生成JSONL文件"""
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            if parsed := parse_log_line(line.strip()):
                fout.write(json.dumps(parsed) + '\n')


def plot():
    with open("log/distill_grpo_log_file.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    data = data[1:]
    # reward mean
    reward_mean = [item["critic_rewards_mean"] for item in data]
    plt.plot(reward_mean)
    plt.xlabel("steps")
    plt.ylabel("mean reward")
    plt.savefig("pictures/reawrd_mean.png", dpi=600)
    plt.show()
    # response_mean
    plt.figure()
    response_length_mean = [item["response_length_mean"] for item in data]
    plt.plot(response_length_mean)
    plt.xlabel("steps")
    plt.ylabel("mean response length")
    plt.savefig("pictures/response_length_mean.png", dpi=600)
    plt.show()
    # actor_kl_loss
    plt.figure()
    actor_kl_loss = [item["actor_kl_loss"] for item in data]
    plt.plot(actor_kl_loss)
    plt.xlabel("steps")
    plt.ylabel("kl loss")
    plt.savefig("pictures/actor_kl_loss.png", dpi=600)
    plt.show()
    # critic_advantages_mean
    plt.figure()
    critic_advantages_mean = [item["critic_advantages_mean"] for item in data]
    plt.plot(critic_advantages_mean)
    plt.xlabel("steps")
    plt.ylabel("mean advantages")
    plt.savefig("pictures/critic_advantages_mean.png", dpi=600)
    plt.show()


# 使用示例
generate_jsonl('log/distill_grpo_log_file.txt', 'log/distill_grpo_log_file.jsonl')
plot()

