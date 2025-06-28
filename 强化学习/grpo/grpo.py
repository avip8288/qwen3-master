# import unsloth
from datasets import load_from_disk
import re
import torch
from datasets import load_dataset, Dataset, interleave_datasets, concatenate_datasets
from transformers import TrainingArguments, Trainer,DataCollatorForSeq2Seq,AutoTokenizer, AutoModelForCausalLM,Trainer
from peft import LoraConfig, get_peft_model, TaskType
import deepspeed
DS_CONFIG = "ds_z1_no_offload_config.json"
from swanlab.integration.transformers import SwanLabCallback
train_dataset = load_from_disk("train_dataset")
test_dataset = load_from_disk("test_dataset")


# from unsloth import FastLanguageModel, PatchFastRL
# PatchFastRL("GRPO", FastLanguageModel)

# from unsloth import is_bfloat16_supported


tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen3-4B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2"
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

# 获取LoRA模型
# 转换模型
peft_model = get_peft_model(model, config)
peft_model.config.use_cache = False


# model,tokenizer = FastLanguageModel.from_pretrained(
#     # model_name = "sft_model",
#     model_name="/root/autodl-tmp/Qwen/Qwen3-4B",
#     # model_name="/root/autodl-tmp/Qwen/Qwen2.5-1.5B-Instruct",
#     max_seq_length = 2048,   
#     # load_in_4bit = True,     
#     # load_in_8bit = False,    
#     full_finetuning = False,
#     # gpu_memory_utilization = 0.5
#     # device_map=device_map
#     # token = "",      
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 8,           #  LoRA秩，建议值为8,16,32,64,128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 32,  # LoRA alpha值，建议设为rank或rank*2
#     lora_dropout = 0.1, # LoRA dropout，0值经过优化
#     bias = "none",    # 偏置设置，"none"已优化
    
#     # [新特性] "unsloth"模式减少30%显存，可适应2倍大的批次大小
#     use_gradient_checkpointing = "unsloth", #梯度检查点，用于长上下文
#     random_state = 3407,  # 随机种子
#     use_rslora = False,   # 是否使用rank stabilized LoRA
#     loftq_config = None,  # LoftQ配置
# )


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


## Reward functions
def correctness_reward_func(prompts, completions, answer, db_set,  **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    rewards = []
    for r,a,dt in zip(extracted_responses, answer, db_set):
        if dt == "gsm8k":
            if a in r:
                rewards.append(2.0)
            elif r == a:
                rewards.append(2.0)    #严格些，可以提高分数
            else:
                rewards.append(0.0)
        else:
            rewards.append(2.0 if r.lower() == a.strip().lower() else 0.0)
    print("rewards==",rewards)
    return rewards


def int_reward_func(completions, db_set, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    for r,dt in zip(extracted_responses,db_set):
        if dt == "gsm8k":
            rewards.append(2.0 if r.isdigit() else 0.0)
        elif dt == "pubmedqa":
            rewards.append(2.0 if ('yes' in r.lower() or 'no' in r.lower() or 'maybe' in r.lower()) else 0.0)
        else:
            rewards.append(2.0 if ('a' in r.lower() or 'b' in r.lower() or 'c' in r.lower() or 'd' in r.lower()) else 0.0)
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    
    matches = [re.match(pattern, r) for r in responses]
    # extracted_responses = [extract_xml_answer(r) for r in responses]
    # print(f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
    
# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-grpo",
    experiment_name="Qwen3-4B",
    description="使用通义千问Qwen3-4B模型在FreedomIntelligence/medical-o1-reasoning-SFT和BAAI/IndustryInstruction_Health-Medicine数据集上微调。",
    config={
        "model": "Qwen/Qwen3-4B",
        "dataset": "https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT",
        "train_data_number": len(train_dataset),
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    gradient_checkpointing=True,

    # bf16 = is_bfloat16_supported(),
    # fp16 = not is_bfloat16_supported(),
     bf16=True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16, # Increase to 4 for smoother training
    num_generations = 16, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 750,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    deepspeed=DS_CONFIG,
)


 
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset=None,
    callbacks=[swanlab_callback],
)
trainer.train(resume_from_checkpoint=True)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
model.save_lora("grpo_saved_lora")
model.save_pretrained_merged("grpo_model", tokenizer)
