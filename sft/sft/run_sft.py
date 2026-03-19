import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from sft import SFTData, DataCollator
import sys
from tqdm import tqdm
import json
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_size", type=str, default="full", help="128, 256, etc.")
args = parser.parse_args()

os.environ["HF_HOME"] = "/work/nvme/bfdu/mdong1/hf_cache"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"

BATCH_SIZE = 4
lr = 5e-6
EPOCH = 3

if not torch.cuda.is_available():
    print("❌ Error: CUDA is not available")
    sys.exit(1)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
)
model.to(device)
model.gradient_checkpointing_enable()

SAVE_PATH = f"/work/nvme/bfdu/mdong1/assignment5/SFT/sample_{args.sample_size}"
os.makedirs(SAVE_PATH, exist_ok=True)

# path = f"/work/nvme/bfdu/mdong1/assignment5/data/math/subsets/sample_{args.sample_size}/train.jsonl"
path = f"/work/nvme/bfdu/mdong1/assignment5/data/math/filtered_train.jsonl"
if not os.path.exists(path):
    print(f"Error: Data file not found at {path}")
else:
    with open(path, 'r') as f:
        train_data = [json.loads(line) for line in f]

dataset = SFTData(train_data, tokenizer)
collator = DataCollator(tokenizer)

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,      
    shuffle=True,     
    collate_fn=collator  
)

val_path =  "/work/nvme/bfdu/mdong1/assignment5/data/math/val.jsonl"
with open(val_path, 'r') as f:
    val_data = [json.loads(line) for line in f]
val_dataset = SFTData(val_data, tokenizer)
val_collator = DataCollator(tokenizer)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE,      
    shuffle=True,     
    collate_fn=val_collator 
)

if len(train_data) < 1025:
    test_step = 10 
    eval_steps = 20 
else:
    test_step = 400 // BATCH_SIZE
    eval_steps = 800 // BATCH_SIZE

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

all_losses = []
all_accs = []
all_val_losses = []
all_val_steps = []
all_entropy = []

best_val_loss = float('inf')
global_step = 0

model.train()
for epoch in range(EPOCH):
    for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss

        with torch.no_grad():
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 2. 计算 Entropy
            probs = torch.nn.functional.softmax(shift_logits, dim=-1)
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_entropy = -torch.sum(probs * log_probs, dim=-1)
            
            correct_mask = shift_labels != -100
            
            if correct_mask.sum() > 0:
                avg_entropy = token_entropy[correct_mask].mean().item()
            else:
                avg_entropy = 0.0

            # 3. 计算 Accuracy
            preds = shift_logits.argmax(dim=-1)
            if correct_mask.sum() > 0:
                acc = (preds == shift_labels)[correct_mask].float().mean()
            else:
                acc = torch.tensor(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        curr_loss = loss.item()
        curr_acc = acc.item()
        all_losses.append(curr_loss)
        all_accs.append(curr_acc)
        all_entropy.append(avg_entropy)

        if step % test_step == 0:
            tqdm.write(f"EPOCH {epoch+1}: Step {step} | Loss: {curr_loss} | Acc: {curr_acc}")

        if step >= 0 and global_step % eval_steps == 0:
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
        global_step += 1
    
    epoch_path = os.path.join(SAVE_PATH, f"epoch_{epoch+1}_final")
    model.save_pretrained(epoch_path)
    tokenizer.save_pretrained(epoch_path)
    tqdm.write(f"✅ Epoch {epoch+1} saved to {epoch_path}")

metrics_file = os.path.join(SAVE_PATH, "train_metrics.json")

metrics_data = {
    "model_name": MODEL_NAME,
    "total_steps": len(all_losses),
    "losses": all_losses,
    "entropy": all_entropy,
    "accuracies": all_accs,
    "eval": {
        "steps": all_val_steps, 
        "losses": all_val_losses
    }
}

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=4)
