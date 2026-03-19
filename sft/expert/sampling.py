import re
from vllm import SamplingParams
from torch.utils.data import Dataset
import torch

def make_prompt(question):
    system_prompt = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    return f"{system_prompt}\n\nUser: {question}\nAssistant: <think>"

def extract_gt_answer(answer_str):
    """提取 MATH 数据集标准答案中 \boxed{} 里的内容"""
    if not answer_str: return None
    # 匹配最后一个 \boxed{...}，处理可能的嵌套括号
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
    
    lines = text.split('\n')
    return lines[-1].strip() if lines else None

def clean_math_answer(text):
    """MATH 数据集不建议用 clean_num，因为会有分数和根号，只做基础清洗"""
    if text is None: return None
    # 去除 LaTeX 常见的控制符和空格
    text = text.replace(r"\text{", "").replace("}", "")
    text = text.replace(r"\quad", "").replace(" ", "")
    text = text.replace(r"$", "")
    return text.strip()

def sampling(D_d, llm, n_G):
    traj = []
    prompts = [make_prompt(item['problem']) for item in D_d]

    sampling_params = SamplingParams(
        n=n_G,
        temperature=0.7, 
        max_tokens=2048,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        gt_ans = extract_gt_answer(D_d[i]['solution'])
        gt_ans = clean_math_answer(gt_ans)

        for generated_sequence in output.outputs:
            generated_text = generated_sequence.text
            pred_ans = extract_answer_from_model(generated_text)
            pred_ans = clean_math_answer(pred_ans)
            if pred_ans == gt_ans:
                traj.append({
                    "prompt_and_problem": prompts[i],
                    "traj": generated_text
                })
    
    return traj


class ExpertDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        prompt = item['prompt_and_problem']
        response_text = item['traj'] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        max_seq_len = 1024

        input_ids = (prompt_ids + response_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]

        if len(input_ids) == max_seq_len:
            input_ids[-1] = self.tokenizer.eos_token_id
            labels[-1] = self.tokenizer.eos_token_id

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }





