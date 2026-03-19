import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from sft import DataCollator, SFTData
import sys
from tqdm import tqdm
import json
import shutil
import argparse
import random
from vllm import LLM, SamplingParams

from sampling import sampling, ExpertDataset
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=str, default="1024", help="128, 256, etc.")
args = parser.parse_args()

if __name__ == "__main__":

    os.environ["HF_HOME"] = "/work/nvme/bfdu/mdong1/hf_cache"
    # MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
    MODEL_PATH = "/work/nvme/bfdu/mdong1/assignment5/SFT/best_model_no_filter/"

    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 16
    lr = 5e-6
    EPOCH = 3

    if not torch.cuda.is_available():
        print("❌ Error: CUDA is not available")
        sys.exit(1)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollator(tokenizer)

    # Load Model
    device = torch.device("cuda")



    SAVE_PATH = f"/work/nvme/bfdu/mdong1/assignment5/EXPERT/sample_{args.sample_size}"
    os.makedirs(SAVE_PATH, exist_ok=True)

    # path = f"/work/nvme/bfdu/mdong1/assignment5/data/math/subsets/sample_{args.sample_size}/train.jsonl"
    path = f"/work/nvme/bfdu/mdong1/assignment5/data/math/filtered_train.jsonl"
    if not os.path.exists(path):
        print(f"Error: Data file not found at {path}")
    else:
        with open(path, 'r') as f:
            train_data = [json.loads(line) for line in f]
        f.close()


    val_path =  "/work/nvme/bfdu/mdong1/assignment5/data/math/val.jsonl"
    with open(val_path, 'r') as f:
        val_data = [json.loads(line) for line in f]
    val_dataset = SFTData(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,      
        shuffle=True,     
        collate_fn=data_collator 
    )



    n_ei_steps = 5
    n_G = 3
    D = int(args.sample_size)

    all_losses = []
    all_accs = []
    all_val_losses = []
    all_val_steps = []

    best_val_loss = float('inf')
    global_step = 0
    test_step = 50
    eval_step = 100

    for n_ei_step in range(n_ei_steps):
        random.shuffle(train_data)
        D_d = train_data[:D]

        llm = LLM(model=MODEL_PATH, max_model_len=2048, gpu_memory_utilization=0.85)
        G = sampling(D_d=D_d, llm=llm, n_G=n_G)

        # 清理显存
        del llm
        gc.collect(); torch.cuda.empty_cache()

        if len(G) == 0:
            print("⚠️警告：本轮没有收集到任何正确轨迹，跳过微调！")
            continue

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        model.to(device)
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        expert_dataset = ExpertDataset(G, tokenizer)
        loader = DataLoader(
            expert_dataset, 
            batch_size=BATCH_SIZE,      
            shuffle=True,     
            collate_fn=data_collator 
        )

        model.train()
        for epoch in range(EPOCH):
            optimizer.zero_grad()
            for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Step {n_ei_step + 1} | Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(model.device)
                labels = batch['labels'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)

                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss / ACCUMULATION_STEPS
                loss.backward()

                with torch.no_grad():
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    preds = shift_logits.argmax(dim=-1)
                    correct_mask = shift_labels != -100
                    if correct_mask.sum() > 0:
                        acc = (preds == shift_labels)[correct_mask].float().mean()
                    else:
                        acc = torch.tensor(0.0)


                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                curr_loss = loss.item()
                curr_acc = acc.item()
                all_losses.append(curr_loss)
                all_accs.append(curr_acc)

                if step % test_step == 0:
                    tqdm.write(f"EPOCH {epoch+1}: Step {step} | Loss: {curr_loss} | Acc: {curr_acc}")

                if step >= 0 and global_step % eval_step == 0:
                    model.eval()
                    val_loss = 0
                    max_eval_batches = 50
                    
                    with torch.no_grad():
                        for v_step, val_batch in enumerate(val_loader):
                            if not v_step < max_eval_batches:
                                break

                            v_ids = val_batch['input_ids'].to(model.device)
                            v_labels = val_batch['labels'].to(model.device)
                            v_mask = val_batch['attention_mask'].to(model.device)

                            outputs = model(input_ids=v_ids, labels=v_labels, attention_mask=v_mask)
                            val_loss += outputs.loss.item()
                    
                    # avg_val_loss = val_loss / (v_step + 1)
                    actual_v_steps = min(len(val_loader), max_eval_batches)
                    avg_val_loss = val_loss / actual_v_steps
                    tqdm.write(f"====== [VALIDATION] ======= [Epoch {epoch+1} | Step {step}] Validation Loss: {avg_val_loss:.4f}")
                    all_val_losses.append(avg_val_loss)
                    all_val_steps.append(global_step)
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_path = os.path.join(SAVE_PATH, "best_model")
                        
                        if os.path.exists(best_path):
                            shutil.rmtree(best_path)
                        
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        tqdm.write(f"⭐ New Best Model Saved (Loss: {avg_val_loss:.4f})")
                    model.train() 

        n_ei_step_path = os.path.join(SAVE_PATH, f"ei_step_{n_ei_step+1}_final")
        model.save_pretrained(n_ei_step_path)
        tokenizer.save_pretrained(n_ei_step_path)
        MODEL_PATH = n_ei_step_path
        tqdm.write(f"✅ Ei_STEP {n_ei_step+1} saved to {n_ei_step_path}")

        del model
        gc.collect(); torch.cuda.empty_cache()

    metrics_data = {
        "model_name": MODEL_PATH,
        "total_steps": len(all_losses),
        "losses": all_losses,
        "accuracies": all_accs,
        "eval": {
            "steps": all_val_steps, 
            "losses": all_val_losses
        }
    }
    metrics_file = os.path.join(SAVE_PATH, "train_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
