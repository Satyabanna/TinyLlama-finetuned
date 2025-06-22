# Medical Chatbot Training Pipeline

A complete training pipeline for fine-tuning a medical chatbot using TinyLlama-1.1B-Chat-v1.0 on doctor-patient conversation data. Optimized for single GPU training with 4GB VRAM (GTX 1650 tested).

## 🏥 Features

- **Memory Efficient**: 4-bit quantization + LoRA for 4GB VRAM compatibility
- **Medical Domain**: Fine-tuned on doctor-patient conversations from HuggingFace
- **Structured Responses**: Clean, multi-sentence medical advice with assessment, recommendation, and notes
- **Interactive Interface**: Easy testing and deployment
- **Ultra-Optimized Training**: Designed for limited hardware resources
- **Production Ready**: Clean, modular codebase with proper error handling

## 📋 Requirements

- NVIDIA GPU with 4GB+ VRAM (GTX 1650 tested)
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Windows 10/11 (tested on Windows 10.0.26100)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Satyabanna/TinyLlama-finetuned.git
cd slmcursor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements include:
- `transformers>=4.36.0`
- `torch>=2.0.0`
- `peft>=0.7.0`
- `datasets>=2.14.0`
- `wandb>=0.15.0`
- `bitsandbytes>=0.41.0`
- `accelerate>=0.24.0`

### 3. Train the Model

```bash
python train_medical_chatbot_ultra_optimized.py
```

This will:
- Download the medical conversation dataset from HuggingFace (`medical_dialog`)
- Preprocess and format the data with medical context prompts
- Fine-tune TinyLlama-1.1B-Chat-v1.0 with ultra-optimized LoRA settings
- Save the model to `./medical_chatbot_model`
- Log training progress to Weights & Biases (if available)

**Training Time**: ~8 hours on GTX 1650 (4GB VRAM)

### 4. Test the Model

```bash
# Interactive chat mode
python inference_medical_chatbot.py --interactive

# Test with sample medical queries
python inference_medical_chatbot.py --test

# Single query test
python inference_medical_chatbot.py --query "I have chest pain, what should I do?"
```

## 📊 Training Metrics & Results

### Model Performance
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Training Loss**: Improved from ~3.2 to ~2.1 over 626 steps
- **Training Time**: ~8 hours on GTX 1650
- **Memory Usage**: ~3.5GB VRAM during training
- **Dataset Size**: 5,000 samples (ultra-optimized for speed)

### Training Configuration (Ultra-Optimized)
- **LoRA Rank**: 4 (ultra-reduced for memory efficiency)
- **LoRA Alpha**: 8
- **Target Modules**: `["q_proj", "v_proj"]` (minimal for speed)
- **Batch Size**: 1 (effective batch size: 8 with gradient accumulation)
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 1e-3 (higher for faster convergence)
- **Epochs**: 1 (single epoch for ultra-fast training)
- **Max Length**: 128 tokens (ultra-reduced for memory efficiency)

### Memory Optimizations
- **4-bit Quantization**: NF4 with double quantization
- **Mixed Precision**: FP16
- **Gradient Checkpointing**: Enabled
- **Memory Fraction**: 85% GPU memory usage
- **Low CPU Memory Usage**: Enabled

## 📁 Project Structure

```
slmcursor/
├── requirements.txt                           # Dependencies
├── data_preprocessing.py                      # Data loading and formatting
├── train_medical_chatbot_ultra_optimized.py   # Ultra-optimized training pipeline
├── inference_medical_chatbot.py               # Inference and testing
├── README.md                                  # This file
├── medical_chatbot_model/                     # Trained model (after training)
│   ├── adapter_config.json                    # LoRA configuration
│   ├── adapter_model.safetensors              # LoRA weights
│   ├── tokenizer.json                         # Tokenizer
│   ├── tokenizer_config.json                  # Tokenizer config
│   └── checkpoint-626/                        # Latest checkpoint
└── wandb/                                     # Training logs (if using W&B)
```

## 🔧 Model Architecture

### Base Model: TinyLlama-1.1B-Chat-v1.0
- **Parameters**: 1.1B
- **Context Length**: 128 tokens (ultra-optimized for 4GB VRAM)
- **Architecture**: Transformer with chat-tuning
- **Vocabulary Size**: 32,000 tokens

### Fine-tuning Strategy
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank**: 4 (ultra-reduced for memory efficiency)
- **Alpha**: 8
- **Target Modules**: Query and Value projections only
- **Quantization**: 4-bit (NF4) with double quantization
- **Trainable Parameters**: ~2.6M (0.24% of base model)

## 📝 Data Format

### Input Format
```
<bos>
Patient: [patient question/symptoms]
Doctor: [doctor response]
<eos>
```

### Output Structure
The model generates structured medical responses with:
- **Assessment**: Initial medical evaluation
- **Recommendation**: Treatment suggestions
- **Additional Notes**: Important considerations
- **Full Response**: Complete medical advice

## 🎯 Inference Parameters

### Optimized for Medical Responses
```python
generation_config = {
    "temperature": 0.7,        # Balanced creativity
    "top_p": 0.9,             # Nucleus sampling
    "top_k": 50,              # Top-k sampling
    "repetition_penalty": 1.1, # Prevent repetition
    "max_length": 256,         # Response length
    "no_repeat_ngram_size": 3  # Prevent 3-gram repetition
}
```

## 🔍 Usage Examples

### Training with Custom Data
```python
from data_preprocessing import load_and_preprocess_data
from train_medical_chatbot_ultra_optimized import MedicalChatbotTrainerUltraOptimized

# Load your custom dataset
dataset_dict = load_and_preprocess_data(file_path="your_data.parquet")

# Train the model
trainer = MedicalChatbotTrainerUltraOptimized()
trainer.train(dataset_dict, num_epochs=1, max_samples=5000)
```

### Custom Inference
```python
from inference_medical_chatbot import MedicalChatbotInference

# Load trained model
chatbot = MedicalChatbotInference("./medical_chatbot_model")

# Generate response
response = chatbot.generate_response(
    "I have been experiencing chest pain for the past week",
    temperature=0.6,
    max_length=200
)

print(f"Assessment: {response['assessment']}")
print(f"Recommendation: {response['recommendation']}")
print(f"Additional Notes: {response['additional_notes']}")
```

## ⚠️ Important Notes

### Medical Disclaimer
⚠️ **CRITICAL**: This model is for educational and research purposes only. It should not be used for actual medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.

### Hardware Requirements
- **Minimum**: GTX 1650 (4GB VRAM) - Tested and working
- **Recommended**: RTX 3060+ (8GB+ VRAM) for faster training
- **CPU Fallback**: Available but much slower (not recommended)

### Memory Usage
- **Training**: ~3.5GB VRAM (optimized for 4GB cards)
- **Inference**: ~2GB VRAM
- **Model Size**: ~700MB (quantized)
- **LoRA Adapter**: ~15MB

### Known Issues & Solutions
1. **Tokenizer Vocabulary Mismatch**: If you encounter vocabulary size errors, the model will automatically fall back to the base model
2. **CUDA Memory Errors**: The ultra-optimized script is designed to work within 4GB VRAM limits
3. **Slow Training**: This is expected on GTX 1650 - the ultra-optimized version reduces training time from hundreds of hours to ~8 hours

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - The ultra-optimized script should handle this automatically
   - If issues persist, reduce `max_samples` in training script

2. **Model Loading Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check CUDA version compatibility
   - Verify model path exists

3. **Poor Response Quality**
   - The model is optimized for speed over quality due to hardware constraints
   - For better quality, retrain with more epochs or larger dataset

### Performance Tips

1. **Faster Training**
   - Use SSD storage
   - Close other GPU applications
   - Ensure adequate system RAM (8GB+)

2. **Better Responses**
   - Lower temperature (0.5-0.7)
   - Increase repetition penalty
   - Use longer max_length

## 📈 Monitoring Training

### Wandb Integration
The training script automatically logs to Weights & Biases if available:
- Loss curves
- Learning rate schedule
- GPU memory usage
- Training metrics
- Model checkpoints

### Local Logging
All training progress is logged to console with detailed metrics:
- Step-by-step loss
- Memory usage
- Training speed
- Validation metrics

## 🎯 Training Results Summary

### Performance Metrics
- **Training Loss**: Reduced from ~3.2 to ~2.1
- **Training Steps**: 626 steps completed
- **Training Time**: ~8 hours on GTX 1650
- **Memory Efficiency**: ~3.5GB VRAM usage
- **Model Convergence**: Good loss improvement observed

### Hardware Performance
- **GPU**: NVIDIA GTX 1650 (4GB VRAM)
- **CPU**: Windows 10.0.26100
- **Memory**: Optimized for 4GB VRAM constraint
- **Storage**: Local SSD recommended

## 🤝 Contributing

Feel free to submit issues and enhancement requests! This project is designed for educational purposes and hardware-constrained environments.

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your local regulations regarding medical AI systems.

---

**Note**: This is a research implementation optimized for limited hardware resources. For production medical applications, additional safety measures, validation, regulatory compliance, and higher-quality training data are required. 
