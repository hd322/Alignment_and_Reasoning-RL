from torch.utils.data import Dataset
import torch
import re

def make_prompt(question):
    system_prompt = (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "Be extremely careful about traps!"
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
        "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    return f"{system_prompt}\n\nUser: {question}\nAssistant: <think>"

class SFTData(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        # prompt = make_prompt(item['question'])

        # reasoning, final_ans = item['answer'].split('####')
        # response_text = f"{reasoning.strip()}\n</think>\n<answer>\n{final_ans.strip()}\n</answer>{self.tokenizer.eos_token}"
        question = item['problem'] # MATH数据集的字段通常是 problem
        solution = item['solution'] # MATH数据集的字段通常是 solution


        # ans_match = re.findall(r'\\boxed\{(.*)\}', solution)
        ans_match = re.findall(r'\\boxed\{(.*?)\}', solution)
        if ans_match:
            final_ans = ans_match[-1].strip()
        else:
            final_ans = ""
        reasoning = solution.strip()

        prompt = make_prompt(question)
        # response_text = f"{reasoning}\n</think>\n<answer>\n{final_ans}\n</answer>{self.tokenizer.eos_token}"
        response_text = f"{reasoning}\n</think>\n<answer>\n\\boxed{{{final_ans}}}\n</answer>{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # # 在 SFTData 的 __getitem__ 结尾处或 encode 处
        # prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False, max_length=1024, truncation=True)
        # response_ids = self.tokenizer.encode(response_text, add_special_tokens=False, max_length=1024, truncation=True)

        max_seq_len = 1024

        # input_ids = prompt_ids + response_ids
        # labels = [-100] * len(prompt_ids) + response_ids

        input_ids = (prompt_ids + response_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + response_ids)[:max_seq_len]
        
        if len(input_ids) == max_seq_len:
            input_ids[-1] = self.tokenizer.eos_token_id
            labels[-1] = self.tokenizer.eos_token_id

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):

        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        batch_input_ids = []
        batch_labels = []
        batch_mask = []

        for i_ids, l_ids in zip(input_ids, labels):

            padding_len = max_len - len(i_ids)

            batch_input_ids.append(torch.cat([
                i_ids, 
                torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ]))

            batch_labels.append(torch.cat([
                l_ids,
                torch.full((padding_len,), -100, dtype=torch.long)
            ]))

            batch_mask.append(torch.cat([
                torch.ones(len(i_ids), dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ]))

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_mask)
        }
