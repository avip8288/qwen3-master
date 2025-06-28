import os
from numpy.core.multiarray import _reconstruct
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import deepspeed
from trl import DPOTrainer,DPOConfig
from peft import LoraConfig, TaskType, get_peft_model
from swanlab.integration.transformers import SwanLabCallback

DS_CONFIG = "ds_z2_offload_config.json"


model_name = "/root/autodl-tmp/merged_model_finetune_1_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)


device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    use_cache=False
)
model.enable_input_require_grads() 

ref_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/merged_model_finetune_1_7b",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation="flash_attention_2",
    use_cache=False
)


train_data = load_dataset("json", data_files="train_preference_data.json")['train']



swanlab_callback = SwanLabCallback(
    project="Qwen3-1.7B-dpo",
    experiment_name="Qwen3-1.7B-dpo",
    description="使用通义千问Qwen3-1.7B模型在nuriyev/medical-question-answering-rl-labeled-qwen-0.5B-binarized和Morefreedai/medical-dpo-v1数据集上微调。",
    config={
        "model": "Qwen/Qwen3-1.7B",
        "dataset": "https://huggingface.co/datasets/nuriyev/medical-question-answering-rl-labeled-qwen-0.5B-binarized",
        "train_data_number": len(train_data),
        "lora_rank": 128,
        "lora_alpha": 128,
        "lora_dropout": 0.1,
    }
)

# Train the model
training_args = DPOConfig(
    output_dir="./zero-dpo",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=5,
    save_steps=500,
    learning_rate=1e-4,
    logging_first_step=5,
    max_grad_norm=1.0,
    beta=0.5    #0.1--0.5,越小越稳定
)


trainer = DPOTrainer(
     model=model,
     ref_model=ref_model, # 如果是相对优化，可以提供一个 reference model,如果用了peft，就设置为None
     train_dataset=train_data,
     eval_dataset=None,
     tokenizer=tokenizer,
     callbacks=[swanlab_callback],
     args=training_args
)
  
trainer.train()
trainer.save_model('./zero-dpo')
trainer.save_state()
