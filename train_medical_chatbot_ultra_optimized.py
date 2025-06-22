import os
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import DatasetDict
import wandb
from data_preprocessing import load_and_preprocess_data, add_medical_context_prompts

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbotTrainerUltraOptimized:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Enhanced CUDA setup
        self._setup_cuda()
    
    def _setup_cuda(self):
        """Enhanced CUDA setup for optimal GPU utilization."""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Set CUDA device
            torch.cuda.set_device(0)
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            cuda_version = torch.version.cuda
            
            logger.info(f"✅ CUDA available: {gpu_name}")
            logger.info(f"✅ GPU Memory: {gpu_memory:.2f} GB")
            logger.info(f"✅ CUDA Version: {cuda_version}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.85)
                
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU (will be very slow)")
    
    def setup_tokenizer(self):
        """Initialize and configure the tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add special tokens if not present
        special_tokens = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>"
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def setup_model(self, use_4bit: bool = True, use_lora: bool = True):
        """Initialize and configure the model with quantization and LoRA."""
        logger.info(f"Loading model from {self.model_name}")
        
        # Load model with 4-bit quantization for memory efficiency
        if use_4bit:
            from transformers.utils.quantization_config import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        
        # Prepare model for k-bit training
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Add LoRA configuration for efficient fine-tuning
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=4,  # Ultra-reduced rank for maximum memory efficiency
                lora_alpha=8,  # Ultra-reduced alpha
                lora_dropout=0.1,
                target_modules=[
                    "q_proj",
                    "v_proj",
                ],  # Minimal target modules
                bias="none",
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Resize token embeddings to match tokenizer
        if self.tokenizer is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move model to device explicitly
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            logger.info("✅ Model moved to CUDA")
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
    
    def tokenize_function(self, examples):
        """Tokenize the text data."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer() first.")
        
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,  # Ultra-reduced max length for maximum memory efficiency
            return_tensors=None,
        )
    
    def prepare_dataset(self, dataset_dict: DatasetDict, max_samples: int = 5000) -> DatasetDict:
        """Prepare and tokenize the dataset with limited samples for ultra-fast training."""
        logger.info(f"Preparing ultra-optimized dataset with max {max_samples} samples...")
        
        # Take only a subset of the data for ultra-fast training
        train_subset = dataset_dict['train'].select(range(min(max_samples, len(dataset_dict['train']))))
        val_subset = dataset_dict['validation'].select(range(min(max_samples // 10, len(dataset_dict['validation']))))
        
        logger.info(f"Using {len(train_subset)} training samples and {len(val_subset)} validation samples")
        
        # Add medical context prompts
        train_subset = add_medical_context_prompts(train_subset)
        
        # Tokenize datasets
        tokenized_train = train_subset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_subset.column_names,
        )
        
        tokenized_val = val_subset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_subset.column_names,
        )
        
        return DatasetDict({
            'train': tokenized_train,
            'validation': tokenized_val
        })
    
    def train(self, 
              dataset_dict: DatasetDict,
              output_dir: str = "./medical_chatbot_model",
              num_epochs: int = 1,  # Single epoch for ultra-fast training
              batch_size: int = 1,  # Minimal batch size
              gradient_accumulation_steps: int = 8,  # Reduced for faster updates
              learning_rate: float = 1e-3,  # Higher learning rate for faster convergence
              warmup_steps: int = 10,  # Minimal warmup
              save_steps: int = 500,  # Save more frequently
              eval_steps: int = 500,  # Evaluate more frequently
              logging_steps: int = 5,  # Log very frequently
              max_samples: int = 5000):  # Limit dataset size
        """Train the medical chatbot model with ultra-optimized settings for 4GB VRAM."""
        
        # Setup model and tokenizer
        self.setup_tokenizer()
        self.setup_model(use_4bit=True, use_lora=True)
        
        # Prepare dataset with limited samples
        tokenized_datasets = self.prepare_dataset(dataset_dict, max_samples)
        
        # Data collator
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer() first.")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Training arguments ultra-optimized for 4GB VRAM
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_strategy="steps",
            fp16=True,  # Use mixed precision
            dataloader_pin_memory=False,  # Save memory
            remove_unused_columns=False,
            report_to="wandb" if wandb.run else None,
            run_name="medical-chatbot-ultra-optimized",
            gradient_checkpointing=True,  # Enable gradient checkpointing
            optim="adamw_torch",  # Use PyTorch optimizer
            lr_scheduler_type="cosine",  # Use cosine scheduler
            dataloader_num_workers=0,  # Disable multiprocessing for Windows
            ddp_find_unused_parameters=False,  # Optimize for single GPU
            max_grad_norm=1.0,  # Gradient clipping
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting ultra-optimized training...")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Total training steps: {len(tokenized_datasets['train']) // (batch_size * gradient_accumulation_steps) * num_epochs}")
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter separately
        if hasattr(self.model, 'save_pretrained') and self.model is not None:
            self.model.save_pretrained(output_dir)
        
        logger.info("Ultra-optimized training completed!")
        return trainer

def main():
    """Main training function with ultra-optimized settings."""
    # Initialize wandb (optional)
    try:
        wandb.init(project="medical-chatbot", name="tinyllama-ultra-optimized")
    except:
        logger.info("Wandb not available, continuing without logging")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing dataset...")
    dataset_dict = load_and_preprocess_data()
    
    # Initialize trainer with ultra-optimized settings
    trainer = MedicalChatbotTrainerUltraOptimized()
    
    # Train the model with ultra-optimized parameters
    trainer.train(
        dataset_dict=dataset_dict,
        output_dir="./medical_chatbot_model",
        num_epochs=1,  # Single epoch
        batch_size=1,  # Minimal batch size
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=1e-3,  # Higher learning rate
        warmup_steps=10,
        save_steps=500,
        eval_steps=500,
        logging_steps=5,
        max_samples=5000  # Use only 5000 samples
    )
    
    logger.info("Ultra-optimized training pipeline completed successfully!")

if __name__ == "__main__":
    main() 