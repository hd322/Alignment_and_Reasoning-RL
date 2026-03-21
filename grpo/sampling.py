import re
from vllm import SamplingParams
from torch.utils.data import Dataset
import torch
import sympy
from sympy.parsing.sympy_parser import parse_expr
from clean_math_answer import clean_math_answer
from drgrpo_grader import grade

def make_prompt(question):
    system_prompt = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    return f"{system_prompt}\n\nUser: {question}\nAssistant: <think>"

# def professional_clean(expr):
#     """提取你提供代码中的归一化逻辑"""
#     if expr is None: return ""
#     # 移除 \text{}, %, $, 单元单位等
#     expr = re.sub(r"\\text\{(.+?)\}", r"\1", str(expr))
#     expr = expr.replace("\\%", "").replace("\\$", "").replace("$", "").replace("%", "")
#     # 常见单位过滤
#     for unit in ["degree", "cm", "meter", "mile", "second", "hour", "day", "inch"]:
#         expr = re.sub(f"{unit}(es)?(s)?", "", expr)
#     expr = expr.replace(" ", "").lower()
#     return expr
# def professional_clean(expr):
#     """提取归一化逻辑，增加对 LaTeX 分数的预处理"""
#     if expr is None: return ""
#     expr = str(expr)
    
#     # 移除 \text{}, %, $, 单元单位等
#     expr = re.sub(r"\\text\{(.+?)\}", r"\1", expr)
#     expr = expr.replace("\\%", "").replace("\\$", "").replace("$", "").replace("%", "")
    
#     # 【新增】处理 LaTeX 分数，统一转换为 (a)/(b)
#     # 1. 先把 \frac56 这种没有括号的，变成 \frac{5}{6}
#     expr = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", expr)
#     # 2. 再把 \frac{a}{b} 转换为 (a)/(b)
#     expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
    
#     # 常见单位过滤
#     for unit in ["degree", "cm", "meter", "mile", "second", "hour", "day", "inch"]:
#         expr = re.sub(f"{unit}(es)?(s)?", "", expr)
        
#     expr = expr.replace(" ", "").lower()
#     return expr
def professional_clean(expr):
    """增强版专业归一化逻辑"""
    if expr is None: return ""
    expr = str(expr)
    
    # 1. 移除首部的变量声明，例如 "x=0", "y = 3", "N=12" -> "0", "3", "12"
    expr = re.sub(r"^[a-zA-Z]\s*=\s*", "", expr)
    
    # 2. 移除 \text{} 和货币/百分号
    expr = re.sub(r"\\text\{(.+?)\}", r"\1", expr)
    expr = expr.replace("\\%", "").replace("\\$", "").replace("$", "").replace("%", "")
    
    # 3. 处理 LaTeX 分数
    expr = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", expr)
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
    
    # 4. 常见单位过滤 (加入了 cup, cups)
    for unit in ["degree", "cm", "meter", "mile", "second", "hour", "day", "inch", "cup"]:
        expr = re.sub(f"{unit}(es)?(s)?", "", expr)
        
    expr = expr.replace(" ", "").lower()
    
    # 5. 处理无序列表 (解决 -2,7 和 7,-2 的问题)
    # 如果包含逗号，且没有括号(说明是个简单的枚举)，将其排序后再拼接
    if "," in expr and not any(c in expr for c in "()[]{}"):
        parts = sorted(expr.split(","))
        expr = ",".join(parts)
        
    return expr

def professional_check(model_str, gt_str):
    """结合 MathD 和 SymPy 的比对"""
    m_norm = professional_clean(model_str)
    g_norm = professional_clean(gt_str)
    
    # 1. 简单字符串比对
    if m_norm == g_norm: return True
    
    # 2. SymPy 符号比对 (处理 648/π 这种)
    try:
        # 简单转换：将 ^ 换成 Python 的 **
        m_sym = parse_expr(m_norm.replace("^", "**"), transformations='all')
        g_sym = parse_expr(g_norm.replace("^", "**"), transformations='all')
        if sympy.simplify(m_sym - g_sym) == 0:
            return True
    except:
        pass
    return False


def extract_gt_answer(answer_str):
    """提取 MATH 数据集标准答案中 \\boxed{} 里的内容"""
    if not answer_str: return None
    # 匹配最后一个 \\boxed{...}，处理可能的嵌套括号
    res = re.findall(r'\\boxed\{(.*)\}', answer_str)
    if res:
        return res[-1].strip()
    return None

def extract_answer_from_model(text):
    full_text = "<think>" + text
    match = re.search(r'<answer>(.*?)</answer>', full_text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        boxed = re.findall(r'\\boxed\{(.*)\}', content)
        if boxed:
            return boxed[-1].strip()
        return content
    
    boxed = re.findall(r'\\boxed\{(.*)\}', text)
    if boxed:
        return boxed[-1].strip()
    return None

# def clean_math_answer(text):
#     """MATH 数据集基础清洗，增加对逗号和 LaTeX 空格的处理"""
#     if text is None: return None
#     text = str(text)
#     # 移除 \text{} 
#     text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
#     # 移除 LaTeX 的各种空格和逗号 (解决 32,\!000 的问题)
#     text = text.replace(r"\quad", "").replace(r"\,", "").replace(r"\!", "")
#     text = text.replace(" ", "").replace(",", "").replace(r"$", "")
#     return text.strip()

# def clean_math_answer(text):
#     if text is None: return None
#     text = str(text)
#     text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
#     text = text.replace(r"\quad", "").replace(r"\,", "").replace(r"\!", "")
    
#     # 只有当逗号两侧都是数字，且逗号后面恰好是 3 位数字时（千分位），才删除逗号
#     # 否则保留逗号（可能是坐标或列表）
#     text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    
#     text = text.replace(" ", "").replace(r"$", "")
#     return text.strip()

# def clean_math_answer(text):
#     if text is None: return None
#     text = str(text)
    
#     # 1. 强力去除常见的后缀单位
#     units = [
#         "years", "year", "units", "unit", "degrees", "degree", 
#         "cm", "cm^2", "inches", "inch", "meters", "sq units"
#     ]
#     for unit in units:
#         # 使用正则，只匹配末尾的单位，防止误删（如 "1inch" -> "1"）
#         text = re.sub(rf"{unit}\s*$", "", text, flags=re.IGNORECASE)

#     # 2. 去除末尾的百分号
#     text = text.replace("%", "")

#     # 3. 去除变量声明 (如 "n=0" -> "0", "x=" -> "")
#     text = re.sub(r"^[a-z]\s*=\s*", "", text, flags=re.IGNORECASE)

#     # 4. 基础 LaTeX 清理
#     text = re.sub(r"\\text\{(.+?)\}", r"\1", text)
#     text = text.replace(r"\quad", "").replace(r"\,", "").replace(r"\!", "").replace(" ", "").replace(r"$", "")
    
#     return text.strip()

def check_format(text):
    """
    检查模型输出是否符合 <think>...</think> <answer>...</answer> 的结构。
    注意：因为你的 prompt 以 <think> 结尾，模型生成的内容应从推理过程开始。
    """
    # 检查关键标签是否存在
    has_think_close = "</think>" in text
    has_answer_open = "<answer>" in text
    has_answer_close = "</answer>" in text
    
    # 检查顺序：</think> 必须在 <answer> 之前
    if has_think_close and has_answer_open:
        correct_order = text.find("</think>") < text.find("<answer>")
    else:
        correct_order = False
        
    if has_think_close and has_answer_open and has_answer_close and correct_order:
        return 1.0
    return 0.0

# def compute_total_reward(text, gt_ans):
#     """
#     综合计算总奖励
#     """
#     # 1. 格式分：建议给一个较小的权重（如 0.1 ~ 0.2），
#     # 这样可以引导模型先学格式，但不会让格式分盖过答案正确的分数。
#     format_score = check_format(text) * 0.2 
    
#     # 2. 答案分：答对给 1.0
#     pred_ans = extract_answer_from_model(text)
#     pred_ans = clean_math_answer(pred_ans)
    
#     answer_score = 0.0
#     if pred_ans is not None and gt_ans is not None:
#         # 【核心修改】：把死板的 == 替换成了更聪明的 professional_check
#         if professional_check(pred_ans, gt_ans):
#             answer_score = 1.0
            
#     return format_score + answer_score, format_score, answer_score

# def compute_total_reward(text, gt_ans):
    
#     format_score = check_format(text)
    
#     # 1. 格式极度不标准，直接重罚，不看答案
#     if format_score == 0:
#         return -1.0, -1.0, 0.0

#     ans_content = extract_answer_from_model(text)
    
#     # 2. 长度惩罚逻辑
#     length_penalty = 0.0
#     if ans_content and len(ans_content.split()) > 10:
#         length_penalty = -0.2
    
#     # 3. 答案比对
#     pred_ans = clean_math_answer(ans_content)
    
#     # 处理 GT 的清洗（防止 GT 也是脏的）
#     clean_gt = clean_math_answer(gt_ans)
    
#     answer_score = 0.0
#     if pred_ans and clean_gt:
#         if professional_check(pred_ans, clean_gt):
#             answer_score = 1.0
            
#     total_r = 0.2 + answer_score + length_penalty
#     return total_r, 0.2, answer_score

# def compute_total_reward(text, gt_ans):
#     """
#     冲刺阶段专用奖励函数：
#     1. 格式正确基准分降低（因为模型已掌握）。
#     2. 答案正确奖励加码（1.0 -> 1.2）。
#     3. 长度惩罚阶梯化（严打小作文）。
#     """
#     # 1. 格式检查
#     format_score = check_format(text)
#     if format_score == 0:
#         return -1.0, -1.0, 0.0 # 格式崩了依然重罚，维持肌肉记忆

#     ans_content = extract_answer_from_model(text)
    
#     # 2. 阶梯化长度惩罚 (更加激进)
#     length_penalty = 0.0
#     if ans_content:
#         word_count = len(ans_content.split())
#         if word_count > 12:
#             length_penalty = -0.4  # 严重啰嗦，扣除近一半奖励
#         elif word_count > 6:
#             length_penalty = -0.15 # 稍显啰嗦
    
#     # 3. 核心答案比对 (双向清洗，对冲脏数据)
#     pred_ans = clean_math_answer(ans_content)
#     clean_gt = clean_math_answer(gt_ans)
    
#     answer_score = 0.0
#     if pred_ans and clean_gt:
#         # 使用你之前的专业比对逻辑（Sympy 等）
#         if professional_check(pred_ans, clean_gt):
#             answer_score = 1.2  # 提高正确答案的权重，拉大差距
            
#     # 4. 总奖励计算
#     # 格式基准分从 0.2 降为 0.1，把梯度空间留给 Answer
#     total_r = 0.1 + answer_score + length_penalty
    
#     # 这里的返回为了兼容你的日志系统，保持 (total, format, answer) 的结构
#     return total_r, 0.1, answer_score

def compute_total_reward(text, gt_ans):
    # 1. 检查基础格式
    has_format = "</think>" in text and "<answer>" in text and "</answer>" in text
    if not has_format:
        return -1.0, -1.0, 0.0  # 格式不对依然重罚

    # 2. 提取模型答案
    try:
        ans_start = text.find("<answer>") + len("<answer>")
        ans_end = text.find("</answer>")
        ans_content = text[ans_start:ans_end].strip()
    except:
        ans_content = ""

    # 3. 铁腕长度惩罚 (防啰嗦)
    length_penalty = 0.0
    if ans_content:
        word_count = len(ans_content.split())
        if word_count > 12:
            length_penalty = -0.4
        elif word_count > 6:
            length_penalty = -0.15

    # 4. 核心：调用超强 grade 函数判题
    is_correct = False
    if ans_content and gt_ans:
        try:
            # fast=False 会开启高召回率模式，包含 sympy 和 math_verify
            # 它内部会自动处理 gt_ans 自带的 \boxed 或者 \text 等脏数据
            is_correct = grade(ans_content, gt_ans, fast=False)
        except Exception as e:
            is_correct = False
            
    answer_score = 1.2 if is_correct else 0.0
    
    # 5. 计算总分
    total_r = 0.1 + answer_score + length_penalty
    
    return total_r, 0.1, answer_score


def sampling(D_d, llm, n_G):
    traj = []
    prompts = [make_prompt(item['problem']) for item in D_d]

    sampling_params = SamplingParams(
        n=n_G,
        temperature=0.8, 
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True, # 确保包含停止词，方便格式检查
        logprobs=1  
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        group_rewards = []
        group_data = []

        # gt_ans = extract_gt_answer(D_d[i]['solution'])
        # gt_ans = clean_math_answer(gt_ans)

        raw_solution = D_d[i]['solution']

        for generated_sequence in output.outputs:
            generated_text = generated_sequence.text
            
            # --- 使用新的奖励计算逻辑 ---
            # total_r, fmt_r, ans_r = compute_total_reward(generated_text, gt_ans)
            total_r, fmt_r, ans_r = compute_total_reward(generated_text, raw_solution)
            
            group_rewards.append(total_r)

            gen_token_ids = generated_sequence.token_ids
            gen_logprobs_dicts = generated_sequence.logprobs

            extracted_log_probs = []
            for token_id, logprob_dict in zip(gen_token_ids, gen_logprobs_dicts):
                extracted_log_probs.append(logprob_dict[token_id].logprob)

            group_data.append({
                'prompt': prompts[i],
                'completion': generated_text,
                'reward': total_r, # 这里存入总分用于 advantage 计算
                'format_reward': fmt_r, # 存入这些用于后续统计（可选）
                'answer_reward': ans_r,
                'old_log_probs': extracted_log_probs
            })
            
        # --- 优势函数计算 (GRPO 核心) ---
        rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()

        eps = 1e-8
        # 计算该组内每个样本的 Advantage
        advantages = (rewards_tensor - mean_r) / (std_r + eps)

        for j in range(n_G):
            group_data[j]['advantage'] = advantages[j].item()
            traj.append(group_data[j])

    return traj

# def sampling(D_d, llm, n_G):
#     traj = []
#     prompts = [make_prompt(item['problem']) for item in D_d]

#     sampling_params = SamplingParams(
#         n=n_G,
#         temperature=1.0, 
#         max_tokens=1024,
#         stop=["</answer>"],
#         include_stop_str_in_output=True,
#         logprobs=1  
#     )

#     outputs = llm.generate(prompts, sampling_params)

    # for i, output in enumerate(outputs):
    #     group_rewards = []
    #     group_data = []

    #     gt_ans = extract_gt_answer(D_d[i]['solution'])
    #     gt_ans = clean_math_answer(gt_ans)

    #     for generated_sequence in output.outputs:
    #         generated_text = generated_sequence.text
    #         pred_ans = extract_answer_from_model(generated_text)
    #         pred_ans = clean_math_answer(pred_ans)

    #         reward = 1.0 if pred_ans == gt_ans else 0.0
    #         group_rewards.append(reward)

    #         gen_token_ids = generated_sequence.token_ids
    #         gen_logprobs_dicts = generated_sequence.logprobs

    #         extracted_log_probs = []
    #         for token_id, logprob_dict in zip(gen_token_ids, gen_logprobs_dicts):
    #             extracted_log_probs.append(logprob_dict[token_id].logprob)

    #         group_data.append({
    #             'prompt': prompts[i],
    #             'completion': generated_text,
    #             'reward': reward,
    #             'old_log_probs': extracted_log_probs
    #         })
    #     rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
    #     mean_r = rewards_tensor.mean()
    #     std_r = rewards_tensor.std()

    #     eps = 1e-8
    #     advantages = (rewards_tensor - mean_r) / (std_r + eps)

    #     for j in range(n_G):
    #         group_data[j]['advantage'] = advantages[j].item()
    #         traj.append(group_data[j])

    # return traj


class GRPODataset(Dataset): # 传进来的是G_traj
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        prompt = item['prompt']
        response_text = item['completion'] + self.tokenizer.eos_token
        advantage = item['advantage']
        old_log_probs_list = item['old_log_probs'] 

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        max_seq_len = 1024

        input_ids = (prompt_ids + response_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]

        # aligned_old_log_probs = [0.0] * len(prompt_ids) + old_log_probs_list
        aligned_old_log_probs = [0.0] * (len(prompt_ids) - 1) + old_log_probs_list + [0.0]

        target_len = len(prompt_ids) + len(response_ids)
        if len(aligned_old_log_probs) < target_len:
            aligned_old_log_probs += [0.0] * (target_len - len(aligned_old_log_probs))
        
        aligned_old_log_probs = aligned_old_log_probs[:max_seq_len]

        if len(input_ids) == max_seq_len:
            input_ids[-1] = self.tokenizer.eos_token_id
            labels[-1] = self.tokenizer.eos_token_id
            aligned_old_log_probs[-1] = 0.0
        
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "advantage": torch.tensor(advantage, dtype=torch.float32),
            "old_log_probs": torch.tensor(aligned_old_log_probs, dtype=torch.float32)
        }