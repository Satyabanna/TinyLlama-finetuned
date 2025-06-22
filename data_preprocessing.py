import pandas as pd
from datasets import Dataset, DatasetDict
import re
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text or pd.isna(text):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', str(text).strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
    
    return text

def format_conversation(patient_text: str, doctor_text: str) -> str:
    """Format conversation in the required structure."""
    patient_clean = clean_text(patient_text)
    doctor_clean = clean_text(doctor_text)
    
    if not patient_clean or not doctor_clean:
        return None
    
    # Format: <bos>\nPatient: ...\nDoctor: ...\n<eos>
    conversation = f"<bos>\nPatient: {patient_clean}\nDoctor: {doctor_clean}\n<eos>"
    return conversation

def load_and_preprocess_data(file_path: str = None, huggingface_dataset: str = "ruslanmv/ai-medical-chatbot") -> DatasetDict:
    """
    Load and preprocess the medical conversation dataset.
    
    Args:
        file_path: Path to local parquet file (optional)
        huggingface_dataset: HuggingFace dataset name
    
    Returns:
        DatasetDict with train/validation splits
    """
    try:
        if file_path:
            # Load local parquet file
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded local dataset with {len(df)} conversations")
        else:
            # Load from HuggingFace
            from datasets import load_dataset
            dataset = load_dataset(huggingface_dataset)
            df = dataset['train'].to_pandas()
            logger.info(f"Loaded HuggingFace dataset with {len(df)} conversations")
        
        # Ensure required columns exist
        required_cols = ['Patient', 'Doctor']
        if not all(col in df.columns for col in required_cols):
            # Try alternative column names
            alt_cols = {
                'patient': 'Patient',
                'doctor': 'Doctor',
                'question': 'Patient',
                'answer': 'Doctor',
                'input': 'Patient',
                'output': 'Doctor'
            }
            
            for alt_col, target_col in alt_cols.items():
                if alt_col in df.columns:
                    df[target_col] = df[alt_col]
        
        # Filter out rows with missing data
        df = df.dropna(subset=['Patient', 'Doctor'])
        df = df[df['Patient'].astype(str).str.len() > 10]  # Minimum patient text length
        df = df[df['Doctor'].astype(str).str.len() > 20]   # Minimum doctor text length
        
        # Format conversations
        df['text'] = df.apply(lambda row: format_conversation(row['Patient'], row['Doctor']), axis=1)
        df = df.dropna(subset=['text'])
        
        # Create train/validation split
        train_size = int(0.9 * len(df))
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['text']])
        val_dataset = Dataset.from_pandas(val_df[['text']])
        
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        logger.info(f"Preprocessed dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        return dataset_dict
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def create_medical_prompt_template() -> str:
    """Create a medical conversation prompt template."""
    return """<bos>
Patient: {patient_input}
Doctor: """

def add_medical_context_prompts(dataset: Dataset, num_samples: int = 100) -> Dataset:
    """
    Add medical context prompts to improve model's medical knowledge.
    
    Args:
        dataset: Input dataset
        num_samples: Number of medical context samples to add
    
    Returns:
        Enhanced dataset with medical context
    """
    medical_contexts = [
        "Patient: I have been experiencing chest pain for the past week. What could this be?\nDoctor: Chest pain can have various causes, from mild to serious. I need to ask you some questions to better understand your situation. How would you describe the pain - is it sharp, dull, or pressure-like? Does it worsen with activity or breathing?",
        
        "Patient: My blood pressure readings have been high lately. Should I be concerned?\nDoctor: High blood pressure is a common but important health concern. Let me help you understand what this means and what steps we should take. What are your typical readings? Do you have any family history of hypertension?",
        
        "Patient: I've been feeling very tired and weak for the past month. What tests should I get?\nDoctor: Fatigue and weakness can be symptoms of various conditions. To help determine the cause, I'd recommend starting with some basic blood tests including a complete blood count, thyroid function tests, and vitamin D levels. Have you noticed any other symptoms?",
        
        "Patient: I have diabetes and my blood sugar has been difficult to control. What can I do?\nDoctor: Managing diabetes requires a comprehensive approach. Let's review your current treatment plan and lifestyle factors. What medications are you currently taking? How is your diet and exercise routine? Have you been monitoring your blood sugar regularly?",
        
        "Patient: I'm experiencing severe headaches that won't go away. Should I see a specialist?\nDoctor: Persistent severe headaches definitely warrant medical attention. I need to understand more about your symptoms to determine the best course of action. How long have you been experiencing these headaches? Are they accompanied by any other symptoms like vision changes or nausea?"
    ]
    
    # Add medical context samples
    context_texts = [f"<bos>\n{context}\n<eos>" for context in medical_contexts]
    
    # Create additional dataset with medical contexts
    context_dataset = Dataset.from_dict({'text': context_texts})
    
    # Combine with original dataset
    combined_dataset = Dataset.from_dict({
        'text': dataset['text'] + context_texts
    })
    
    logger.info(f"Added {len(medical_contexts)} medical context samples")
    return combined_dataset

if __name__ == "__main__":
    # Test data preprocessing
    dataset_dict = load_and_preprocess_data()
    print(f"Dataset structure: {dataset_dict}")
    print(f"Sample conversation:\n{dataset_dict['train'][0]['text']}") 