# 1. 필수 라이브러리 설치 (Kaggle/Colab 전용)
# !pip install unsloth
# !pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from datasets import load_dataset

# 1. 모델 설정
max_seq_length = 2048 # 컨텍스트 길이
dtype = None # None으로 두면 자동 설정
load_in_4bit = True # 메모리 절약을 위한 4bit 양자화

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-4-2b-it", # Gemma 4 2B 인스트럭트 모델
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. LoRA(Low-Rank Adaptation) 설정 - 의료 도메인 지식 주입을 위한 어댑터
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank (16, 32, 64 중 선택)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. 데이터셋 로드 (preprocess.py로 만든 jsonl 파일)
dataset = load_dataset("json", data_files={"train": "data/train_data.jsonl"}, split="train")

# 4. 학습 설정 (Hyperparameters)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # preprocess에서 만든 필드명
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # 데이터가 660개라면 1~2 에폭 정도에 해당하는 스텝 수
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. 학습 시작
trainer_stats = trainer.train()

# 6. 로컬(Ollama)에서 사용하기 위한 GGUF 변환 및 저장
# 저장 방식: q4_k_m (성능과 용량의 밸런스가 가장 좋음)
model.save_pretrained_gguf("gemma4_medical_final", tokenizer, quantization_method = "q4_k_m")

print("✅ 학습 및 GGUF 변환 완료! gemma4_medical_final.gguf 파일을 다운로드하세요.")