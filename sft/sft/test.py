from vllm import LLM, SamplingParams
import os

# 指向你刚才炼出来的最强模型
MODEL_PATH = "/work/nvme/bfdu/mdong1/assignment5/SFT/best_model"

os.environ["HF_HOME"] = "/work/nvme/bfdu/mdong1/hf_cache"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

llm = LLM(model=MODEL_PATH, trust_remote_code=True)

# 采样参数，保持 temperature 为 0 观察最稳定的逻辑
sampling_params = SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"])

# 构造 Prompt，诱导模型开启 <think> 模式
question = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
prompt = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags.\n\n"
    f"User: {question}\nAssistant: <think>"
)

outputs = llm.generate([prompt], sampling_params)

print("\n" + "="*50)
print(f"Assistant: <think>{outputs[0].outputs[0].text}")
print("="*50)