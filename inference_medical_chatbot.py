import torch
import re
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
from typing import Dict, List, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatbotInference:
    def __init__(self, model_path: str = "./medical_chatbot_model", use_base_model: bool = False):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.use_base_model = use_base_model
        
        logger.info(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Check if trained model exists
        if not self.use_base_model and os.path.exists(self.model_path):
            logger.info("Loading fine-tuned model...")
            self._load_fine_tuned_model()
        else:
            logger.info("Loading base model (TinyLlama-1.1B-Chat-v1.0)...")
            self._load_base_model()
    
    def _load_fine_tuned_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with 4-bit quantization for inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load LoRA adapter if it exists
            try:
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                logger.info("Loaded LoRA adapter successfully")
            except:
                self.model = base_model
                logger.info("No LoRA adapter found, using base model")
            
            self.model.eval()
            logger.info("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            logger.info("Falling back to base model...")
            self._load_base_model()
    
    def _load_base_model(self):
        """Load the base TinyLlama model for testing."""
        try:
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
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
            
            # Load base model with 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Resize token embeddings to match tokenizer
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.eval()
            logger.info("Base model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading base model: {e}")
            raise
    
    def format_prompt(self, patient_input: str) -> str:
        """Format the input prompt for the model."""
        return f"<bos>\nPatient: {patient_input}\nDoctor:"
    
    def clean_response(self, response: str) -> Dict[str, str]:
        """
        Clean and structure the doctor's response into sections.
        
        Args:
            response: Raw model response
            
        Returns:
            Dictionary with structured sections
        """
        # Remove the prompt part and get only the doctor's response
        if "Doctor:" in response:
            doctor_part = response.split("Doctor:")[-1].strip()
        else:
            doctor_part = response.strip()
        
        # Remove any trailing tokens
        doctor_part = re.sub(r'<eos>.*$', '', doctor_part, flags=re.DOTALL)
        doctor_part = doctor_part.strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', doctor_part)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Structure the response
        structured_response = {
            "raw_response": doctor_part,
            "assessment": "",
            "recommendation": "",
            "additional_notes": "",
            "full_response": doctor_part
        }
        
        if len(sentences) >= 2:
            # First sentence as assessment
            structured_response["assessment"] = sentences[0] + "."
            
            # Middle sentences as recommendation
            if len(sentences) >= 3:
                structured_response["recommendation"] = ". ".join(sentences[1:-1]) + "."
            else:
                structured_response["recommendation"] = sentences[1] + "."
            
            # Last sentence as additional notes
            if len(sentences) >= 3:
                structured_response["additional_notes"] = sentences[-1] + "."
        
        elif len(sentences) == 1:
            structured_response["assessment"] = sentences[0] + "."
        
        return structured_response
    
    def generate_response(self, 
                         patient_input: str,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         repetition_penalty: float = 1.1,
                         max_length: int = 256,
                         do_sample: bool = True) -> Dict[str, str]:
        """
        Generate a medical response with optimized parameters.
        
        Args:
            patient_input: Patient's question/symptoms
            temperature: Controls randomness (0.1-1.0)
            top_p: Nucleus sampling parameter (0.1-1.0)
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            max_length: Maximum response length
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary with structured response
        """
        # Format prompt
        prompt = self.format_prompt(patient_input)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generation parameters optimized for medical responses
        generation_config = {
            "max_new_tokens": max_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3,  # Prevent repetition of 3-grams
            "early_stopping": True,
            "length_penalty": 1.0,
            "num_beams": 1,  # Use greedy decoding for medical responses
        }
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean and structure response
        structured_response = self.clean_response(response)
        
        return structured_response
    
    def batch_generate(self, 
                      patient_inputs: List[str],
                      **kwargs) -> List[Dict[str, str]]:
        """Generate responses for multiple patient inputs."""
        responses = []
        for patient_input in patient_inputs:
            response = self.generate_response(patient_input, **kwargs)
            responses.append(response)
        return responses
    
    def interactive_chat(self):
        """Interactive chat interface for testing."""
        model_type = "Base TinyLlama" if self.use_base_model or not os.path.exists(self.model_path) else "Fine-tuned Medical"
        print(f"Medical Chatbot - Interactive Mode ({model_type})")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            patient_input = input("\nPatient: ").strip()
            
            if patient_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not patient_input:
                continue
            
            try:
                response = self.generate_response(patient_input)
                
                print("\nDoctor:")
                print(f"Assessment: {response['assessment']}")
                if response['recommendation']:
                    print(f"Recommendation: {response['recommendation']}")
                if response['additional_notes']:
                    print(f"Additional Notes: {response['additional_notes']}")
                print("-" * 50)
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print("I apologize, but I'm having trouble processing your request. Please try again.")

def test_medical_queries():
    """Test the model with various medical queries."""
    # Check if trained model exists
    if os.path.exists("./medical_chatbot_model"):
        chatbot = MedicalChatbotInference()
        model_type = "Fine-tuned Medical"
    else:
        logger.info("Trained model not found, using base model for testing...")
        chatbot = MedicalChatbotInference(use_base_model=True)
        model_type = "Base TinyLlama"
    
    test_queries = [
        "I have been experiencing chest pain for the past week. What could this be?",
        "My blood pressure readings have been high lately. Should I be concerned?",
        "I've been feeling very tired and weak for the past month. What tests should I get?",
        "I have diabetes and my blood sugar has been difficult to control. What can I do?",
        "I'm experiencing severe headaches that won't go away. Should I see a specialist?"
    ]
    
    print(f"Testing Medical Chatbot with Sample Queries ({model_type})")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 40)
        
        try:
            response = chatbot.generate_response(query)
            
            print("Doctor's Response:")
            print(f"Assessment: {response['assessment']}")
            if response['recommendation']:
                print(f"Recommendation: {response['recommendation']}")
            if response['additional_notes']:
                print(f"Additional Notes: {response['additional_notes']}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 60)

def main():
    """Main function for inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Chatbot Inference")
    parser.add_argument("--model_path", default="./medical_chatbot_model", 
                       help="Path to the trained model")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--test", action="store_true", 
                       help="Run test queries")
    parser.add_argument("--query", type=str, 
                       help="Single query to test")
    parser.add_argument("--base_model", action="store_true",
                       help="Use base model instead of fine-tuned model")
    
    args = parser.parse_args()
    
    if args.test:
        test_medical_queries()
    elif args.interactive:
        chatbot = MedicalChatbotInference(args.model_path, use_base_model=args.base_model)
        chatbot.interactive_chat()
    elif args.query:
        chatbot = MedicalChatbotInference(args.model_path, use_base_model=args.base_model)
        response = chatbot.generate_response(args.query)
        print(f"Query: {args.query}")
        print(f"Response: {response['full_response']}")
    else:
        print("Please specify --interactive, --test, or --query")
        print("\nExamples:")
        print("  python inference_medical_chatbot.py --test")
        print("  python inference_medical_chatbot.py --interactive")
        print("  python inference_medical_chatbot.py --base_model --interactive")

if __name__ == "__main__":
    main() 