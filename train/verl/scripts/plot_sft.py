import re
import matplotlib.pyplot as plt


def extract_loss(log_path):
    """从训练日志中提取loss序列"""
    loss_list = []
    pattern = re.compile(r'train/loss:(\d+\.\d+)')  # 匹配loss值的正则表达式
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    loss = float(match.group(1))
                    loss_list.append(loss)
                except ValueError:
                    print(f"格式异常行：{line.strip()}")
    
    return loss_list

def plot_loss_curve(loss_list):
    """绘制loss变化曲线"""
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(loss_list)), loss_list, 
             color='#2c7fb8', linewidth=1.5)
    
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('pictures/loss_sft.png', dpi=300)
    plt.show()

# 使用示例
if __name__ == "__main__":
    loss_data = extract_loss('log/qwen_32b_sft_full_stage2_v1.txt')
    print(f"共提取到{len(loss_data)}条loss记录")
    plot_loss_curve(loss_data)


