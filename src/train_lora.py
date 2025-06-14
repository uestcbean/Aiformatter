import os
import json
import torch
import logging
import gc
import psutil
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

# 优化后的配置 - 针对RTX 4090 24GB
CONFIG = {
    "MODEL_PATH": os.getenv("MODEL_PATH", "D:/ai/openai/Qwen-7B-Chat"),
    "DATASET_PATH": os.getenv("DATASET_PATH", "data/chatml_augmented_dataset_processed_fixed.jsonl"),
    "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "output/lora_qwen7b_optimized"),
    "LOG_DIR": os.getenv("LOG_DIR", "D:/ai/openai/qwen_lora_project/logs"),
    
    # LoRA配置优化
    "LORA_R": int(os.getenv("LORA_R", "16")),  # 从8增加到16
    "LORA_ALPHA": int(os.getenv("LORA_ALPHA", "32")),
    "LORA_DROPOUT": float(os.getenv("LORA_DROPOUT", "0.05")),  # 降低dropout
    
    # 训练参数优化
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "2e-4")),  # 提高学习率
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "4")),  # 增加批次大小
    "GRADIENT_ACCUMULATION_STEPS": int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2")),  # 减少累积步数
    "NUM_EPOCHS": int(os.getenv("NUM_EPOCHS", "3")),
    "MAX_LENGTH": int(os.getenv("MAX_LENGTH", "512")),  # 增加序列长度
    
    # 显存配置
    "USE_QUANTIZATION": os.getenv("USE_QUANTIZATION", "false").lower() == "true",  # 默认不量化
    "GPU_MEMORY_LIMIT": os.getenv("GPU_MEMORY_LIMIT", "22GB"),  # 预留2GB
}

# 环境优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(8)  # 增加CPU线程数

# 创建输出目录
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logs["gpu_memory_allocated_gb"] = f"{allocated:.2f}"
            logs["gpu_memory_reserved_gb"] = f"{reserved:.2f}"

def setup_logging():
    """设置日志系统"""
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = os.path.join(CONFIG["LOG_DIR"], log_filename)
    
    logger = logging.getLogger('qwen_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_system():
    """验证系统配置"""
    logger.info("=== 系统验证 ===")
    
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        raise FileNotFoundError(f"模型路径不存在: {CONFIG['MODEL_PATH']}")
    if not os.path.exists(CONFIG["DATASET_PATH"]):
        raise FileNotFoundError(f"数据集文件不存在: {CONFIG['DATASET_PATH']}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    logger.info(f"系统内存: {memory_gb:.1f}GB")
    logger.info(f"GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
    logger.info("系统验证通过")

def setup_memory_optimization():
    """优化内存设置"""
    logger.info("=== 内存优化设置 ===")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 对于24GB显卡，可以使用更高比例
        torch.cuda.set_per_process_memory_fraction(0.90)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # 启用cudnn优化
    
    logger.info("内存优化设置完成")

def load_model_and_tokenizer():
    """加载模型和分词器"""
    logger.info("=== 加载分词器 ===")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_PATH"], trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"分词器加载完成")
    
    logger.info("=== 加载模型 ===")
    
    # 根据配置决定是否量化
    if CONFIG["USE_QUANTIZATION"]:
        logger.info("使用8-bit量化")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        torch_dtype = torch.float16
    else:
        logger.info("不使用量化，直接加载16-bit模型")
        quant_config = None
        torch_dtype = torch.float16
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["MODEL_PATH"],
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map="auto",  # 自动设备映射
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_cache=False,  # 训练时关闭cache
        )
        
        # 启用优化
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"模型加载成功 - GPU内存占用: {allocated_gb:.2f}GB")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

def setup_lora(model):
    """设置LoRA配置"""
    logger.info("=== 配置LoRA ===")
    
    # 扩展目标模块，包含更多层
    target_modules = [
        "c_attn", "c_proj",  # 原有的注意力层
        "w1", "w2",          # 原有的MLP层
        "c_fc",              # 全连接层
    ]
    
    lora_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    if CONFIG["USE_QUANTIZATION"]:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    
    # 显示参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = 100 * trainable_params / total_params
    
    logger.info(f"LoRA配置完成")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数比例: {trainable_ratio:.2f}%")
    
    return model

def preprocess_data(tokenizer):
    """优化的数据预处理"""
    logger.info("=== 加载数据集 ===")
    
    try:
        dataset = load_dataset("json", data_files=CONFIG["DATASET_PATH"])
        logger.info(f"原始数据集大小: {len(dataset['train'])}")
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        raise
    
    def preprocess_function(examples):
        """优化的预处理函数"""
        conversations = examples["messages"]
        texts = []
        
        for conversation in conversations:
            formatted_text = ""
            for message in conversation:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            texts.append(formatted_text.strip())
        
        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        
        # 分词
        encoded = tokenizer(
            valid_texts,
            truncation=True,
            max_length=CONFIG["MAX_LENGTH"],
            padding=False,
            return_tensors=None
        )
        
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded
    
    logger.info("=== 预处理数据 ===")
    tokenized_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names,
        num_proc=4,  # 增加并行处理
        desc="Tokenizing"
    )
    
    logger.info(f"预处理后数据集大小: {len(tokenized_dataset)}")
    
    # 数据集统计信息
    lengths = [len(item) for item in tokenized_dataset["input_ids"]]
    logger.info(f"序列长度统计 - 平均: {sum(lengths)/len(lengths):.1f}, 最大: {max(lengths)}, 最小: {min(lengths)}")
    
    return tokenized_dataset

def main():
    """主函数"""
    global logger
    logger = setup_logging()
    
    try:
        logger.info("=== 开始优化训练流程 ===")
        logger.info(f"配置信息: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")
        
        validate_system()
        setup_memory_optimization()
        
        model, tokenizer = load_model_and_tokenizer()
        model = setup_lora(model)
        tokenized_dataset = preprocess_data(tokenizer)
        
        # 优化的数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
        
        # 优化的训练参数
        effective_batch_size = CONFIG["BATCH_SIZE"] * CONFIG["GRADIENT_ACCUMULATION_STEPS"]
        total_steps = (len(tokenized_dataset) // effective_batch_size) * CONFIG["NUM_EPOCHS"]
        
        training_args = TrainingArguments(
            output_dir=CONFIG["OUTPUT_DIR"],
            per_device_train_batch_size=CONFIG["BATCH_SIZE"],
            gradient_accumulation_steps=CONFIG["GRADIENT_ACCUMULATION_STEPS"],
            num_train_epochs=CONFIG["NUM_EPOCHS"],
            learning_rate=CONFIG["LEARNING_RATE"],
            
            # 优化设置
            fp16=True,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,  # 适度增加worker数量
            gradient_checkpointing=True,
            
            # 学习率调度
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            
            # 保存策略 - 减少保存频率
            save_strategy="steps",
            save_steps=max(100, total_steps // 10),  # 最多保存10次
            save_total_limit=3,
            
            # 日志设置
            logging_steps=20,
            logging_dir=CONFIG["LOG_DIR"],
            
            # 其他优化
            remove_unused_columns=False,
            report_to="none",
            skip_memory_metrics=False,
            
            # 优化器设置
            optim="adamw_torch",
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # 性能优化
            bf16=False,  # RTX 4090推荐使用fp16而不是bf16
            tf32=True,   # 启用TF32加速
        )
        
        logger.info(f"训练配置 - 有效批次大小: {effective_batch_size}, 总步数: {total_steps}")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[MemoryMonitorCallback(logger)]
        )
        
        # 检查恢复点
        resume_from_checkpoint = None
        if os.path.exists(CONFIG["OUTPUT_DIR"]):
            checkpoints = [d for d in os.listdir(CONFIG["OUTPUT_DIR"]) if d.startswith("checkpoint")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                resume_from_checkpoint = os.path.join(CONFIG["OUTPUT_DIR"], latest_checkpoint)
                logger.info(f"从检查点恢复训练: {resume_from_checkpoint}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"训练前GPU内存占用: {allocated:.2f}GB")
        
        logger.info("=== 开始训练 ===")
        start_time = datetime.now()
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        logger.info(f"✅ 训练完成! 用时: {training_duration}")
        
        logger.info("=== 保存模型 ===")
        trainer.save_model()
        tokenizer.save_pretrained(CONFIG["OUTPUT_DIR"])
        logger.info("✅ 模型保存完成!")
        
        logger.info("=== 训练流程完成 ===")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()