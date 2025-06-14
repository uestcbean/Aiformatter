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

# Windows环境优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to 1 only for debugging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)  # 限制CPU线程数

# 配置 - 使用环境变量增强可移植性
CONFIG = {
    "MODEL_PATH": os.getenv("MODEL_PATH", "D:/ai/openai/Qwen-7B-Chat"),
    "DATASET_PATH": os.getenv("DATASET_PATH", "data/chatml_augmented_dataset_processed_fixed.jsonl"),
    "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "output/lora_qwen7b_optimized"),
    "LOG_DIR": os.getenv("LOG_DIR", "D:/ai/openai/qwen_lora_project/logs"),
    "LORA_R": int(os.getenv("LORA_R", "8")),
    "LORA_ALPHA": int(os.getenv("LORA_ALPHA", "32")),
    "LORA_DROPOUT": float(os.getenv("LORA_DROPOUT", "0.1")),
    "LEARNING_RATE": float(os.getenv("LEARNING_RATE", "1e-4")),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", "1")),
    "GRADIENT_ACCUMULATION_STEPS": int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
    "NUM_EPOCHS": int(os.getenv("NUM_EPOCHS", "3")),
    "MAX_LENGTH": int(os.getenv("MAX_LENGTH", "128")),  # 增加到更合理的长度
    "GPU_MEMORY_LIMIT": os.getenv("GPU_MEMORY_LIMIT", "20GB"),
}

# 创建输出目录
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
os.makedirs("./offload_temp", exist_ok=True)

# 内存监控回调
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, logger):
        self.logger = logger
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logs["gpu_memory_allocated_gb"] = f"{allocated:.2f}"
            logs["gpu_memory_reserved_gb"] = f"{reserved:.2f}"
            
            # 每100步记录详细内存信息
            if state.global_step % 100 == 0:
                self.logger.info(f"Step {state.global_step} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# 日志设置
def setup_logging():
    """设置日志系统"""
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = os.path.join(CONFIG["LOG_DIR"], log_filename)
    
    logger = logging.getLogger('qwen_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式设置
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 系统验证
def validate_system():
    """验证系统配置和资源"""
    logger.info("=== 系统验证 ===")
    
    # 检查文件
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        raise FileNotFoundError(f"模型路径不存在: {CONFIG['MODEL_PATH']}")
    if not os.path.exists(CONFIG["DATASET_PATH"]):
        raise FileNotFoundError(f"数据集文件不存在: {CONFIG['DATASET_PATH']}")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    # 系统资源信息
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"系统内存: {memory_gb:.1f}GB 总计, {available_gb:.1f}GB 可用")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
        
        if available_gb < 8:
            logger.warning("可用系统内存不足8GB，可能导致加载问题")
    
    logger.info("系统验证通过")

# 内存优化设置
def setup_memory_optimization():
    """设置内存优化"""
    logger.info("=== 内存优化设置 ===")
    
    # 清理缓存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 设置CUDA内存分配策略
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # 使用85%显存
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logger.info("内存优化设置完成")

# 模型设备映射
def create_device_map():
    """为Qwen-7B创建设备映射"""
    device_map = {
        "transformer.wte": 0,
        "transformer.ln_f": 0,
        "lm_head": 0,
    }
    
    # Qwen-7B有32层transformer层
    for i in range(32):
        device_map[f"transformer.h.{i}"] = 0
    
    return device_map

# 加载模型
def load_model_and_tokenizer():
    """加载模型和分词器"""
    logger.info("=== 加载分词器 ===")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_PATH"], trust_remote_code=True)
    
    # 确保正确设置padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 对于Qwen，添加专门的pad token
            tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    
    # 确保pad_token_id正确设置
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"分词器加载完成 - pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    
    logger.info("=== 加载模型 ===")
    
    # 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        llm_int8_skip_modules=["lm_head"]  # 跳过输出层量化
    )
    
    # 设备映射
    device_map = create_device_map()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["MODEL_PATH"],
            trust_remote_code=True,
            quantization_config=quant_config,
            device_map=device_map,
            max_memory={0: CONFIG["GPU_MEMORY_LIMIT"]},
            offload_folder="./offload_temp",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 启用优化
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"模型加载成功 - GPU内存占用: {allocated_gb:.2f}GB")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        # 清理资源
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

# 配置LoRA
def setup_lora(model):
    """设置LoRA配置"""
    logger.info("=== 配置LoRA ===")
    
    lora_config = LoraConfig(
        r=CONFIG["LORA_R"],
        lora_alpha=CONFIG["LORA_ALPHA"],
        lora_dropout=CONFIG["LORA_DROPOUT"],
        bias="none",
        target_modules=["c_attn", "c_proj", "w1", "w2"],  # 扩展目标模块
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 显示可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_ratio = 100 * trainable_params / total_params
    
    logger.info(f"LoRA配置完成")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数比例: {trainable_ratio:.2f}%")
    
    return model

# 数据预处理
def preprocess_data(tokenizer):
    """加载和预处理数据"""
    logger.info("=== 加载数据集 ===")
    
    try:
        dataset = load_dataset("json", data_files=CONFIG["DATASET_PATH"])
        logger.info(f"原始数据集大小: {len(dataset['train'])}")
    except Exception as e:
        logger.error(f"数据集加载失败: {str(e)}")
        raise
    
    def preprocess_function(examples):
        """预处理函数"""
        # 处理ChatML格式的messages
        conversations = examples["messages"]
        
        # 转换为文本格式
        texts = []
        for conversation in conversations:
            # 手动转换messages为对话文本
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
        if len(valid_texts) != len(texts):
            logger.warning(f"过滤了 {len(texts) - len(valid_texts)} 个空文本")
        
        # 分词
        encoded = tokenizer(
            valid_texts,
            truncation=True,
            max_length=CONFIG["MAX_LENGTH"],
            padding=False,  # 使用动态padding
            return_tensors=None
        )
        
        # 设置标签
        encoded["labels"] = encoded["input_ids"].copy()
        
        return encoded
    
    logger.info("=== 预处理数据 ===")
    tokenized_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names,
        num_proc=1,  # Windows上使用单进程避免问题
        desc="Tokenizing"
    )
    
    logger.info(f"预处理后数据集大小: {len(tokenized_dataset)}")
    return tokenized_dataset

# 测试模型加载
def test_model_functionality(model, tokenizer):
    """测试模型基本功能"""
    logger.info("=== 测试模型功能 ===")
    
    try:
        # 测试前向传播
        test_text = "Hello, world!"
        test_input = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model(**test_input)
        
        logger.info("✅ 前向传播测试通过")
        
        # 测试梯度计算 - 添加labels参数
        model.train()
        test_input_with_labels = tokenizer(test_text, return_tensors="pt").to(model.device)
        test_input_with_labels["labels"] = test_input_with_labels["input_ids"].clone()
        
        output = model(**test_input_with_labels)
        loss = output.loss if hasattr(output, 'loss') and output.loss is not None else output.logits.mean()
        loss.backward()
        
        logger.info("✅ 梯度计算测试通过")
        
        # 清理
        model.zero_grad()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 模型功能测试失败: {str(e)}")
        return False

# 清理函数
def cleanup():
    """清理资源"""
    logger.info("清理资源...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("资源清理完成")

# 主训练函数
def main():
    """主函数"""
    global logger
    logger = setup_logging()
    
    try:
        logger.info("=== 开始训练流程 ===")
        logger.info(f"配置信息: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")
        
        # 系统验证
        validate_system()
        
        # 内存优化
        setup_memory_optimization()
        
        # 加载模型和分词器
        model, tokenizer = load_model_and_tokenizer()
        
        # 测试模型功能
        if not test_model_functionality(model, tokenizer):
            raise RuntimeError("模型功能测试失败")
        
        # 配置LoRA
        model = setup_lora(model)
        
        # 预处理数据
        tokenized_dataset = preprocess_data(tokenizer)
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, 
            pad_to_multiple_of=8, 
            return_tensors="pt",
            padding=True  # 动态padding
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=CONFIG["OUTPUT_DIR"],
            per_device_train_batch_size=CONFIG["BATCH_SIZE"],
            gradient_accumulation_steps=CONFIG["GRADIENT_ACCUMULATION_STEPS"],
            num_train_epochs=CONFIG["NUM_EPOCHS"],
            learning_rate=CONFIG["LEARNING_RATE"],
            
            # 优化设置
            fp16=True,
            dataloader_pin_memory=False,  # Windows优化
            dataloader_num_workers=0,     # Windows优化
            gradient_checkpointing=True,
            
            # 保存策略
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            
            # 日志设置
            logging_steps=10,
            logging_dir=CONFIG["LOG_DIR"],
            
            # 其他设置
            remove_unused_columns=False,
            report_to="none",
            skip_memory_metrics=False,  # 启用内存监控
            
            # 训练优化
            warmup_steps=50,
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
        )
        
        # 创建训练器
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
        
        # 训练前内存状态
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"训练前GPU内存占用: {allocated:.2f}GB")
        
        # 开始训练
        logger.info("=== 开始训练 ===")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("✅ 训练完成!")
        
        # 保存模型
        logger.info("=== 保存模型 ===")
        trainer.save_model()
        tokenizer.save_pretrained(CONFIG["OUTPUT_DIR"])
        logger.info("✅ 模型保存完成!")
        
        # 最终内存状态
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"训练后GPU内存占用: {allocated:.2f}GB")
        
        logger.info("=== 训练流程完成 ===")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise
    finally:
        cleanup()

if __name__ == "__main__":
    main()