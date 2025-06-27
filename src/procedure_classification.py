import sys
import re
from pathlib import Path
import gc
import json
from typing import Any
import pandas as pd
from tqdm import tqdm
import getpass
import torch
import os
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
from transformers import pipeline, BitsAndBytesConfig
from huggingface_hub import login


def authenticate_huggingface():
    """Authenticate with Hugging Face if not already authenticated."""
    try:
        # Try to load the model without authentication first
        from huggingface_hub import HfApi
        api = HfApi()
        api.model_info("google/medgemma-27b-text-it")
        print("Already authenticated with Hugging Face!")
        return True
    except Exception:
        print("Authentication required for MedGemma model.")
        
        # Check for environment variable first
        token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        if token:
            print("Using token from environment variable...")
            login(token=token)
            return True
        
        # If no environment variable, prompt for token
        print("Please enter your Hugging Face token:")
        print("(Get it from: https://huggingface.co/settings/tokens)")
        token = getpass.getpass("Token: ")
        
        try:
            login(token=token)
            print("Successfully authenticated!")
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            print("\nPlease ensure:")
            print("1. You have requested access to google/medgemma-27b-text-it")
            print("2. Your access has been approved")
            print("3. Your token is valid")
            return False


def create_prompt(task_config: dict, data: dict[str, Any]) -> list[dict[str, str]]:
    """
    Create a prompt based on task configuration and input data.
    
    Args:
        task_config: Configuration for the current task
        data: Dictionary containing data to substitute in the prompt
        
    Returns:
        Messages list for the LLM
    """
    # Format the user prompt with provided data
    user_prompt = task_config["user_prompt"].format(**data)
    thinking_mode = task_config.get("thinking_mode", False)
    
    if thinking_mode:
        sys_prompt = "SYSTEM INSTRUCTION: think silently if needed. " + task_config["system_prompt"] + " Think in steps, be consize and direct. You must give a final answer."
    else:
        sys_prompt = task_config["system_prompt"]
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]
    
    return messages


class ProcedureClassifier:
    """
    A configurable classifier for medical procedure descriptions using MedGemma.
    Processes tasks sequentially based on JSON configuration.
    """
    
    def __init__(
        self, 
        model_id: str = "google/medgemma-27b-text-it", 
        quantization_bits: int | None = None,
    ):
        """
        Initialize the classifier with the specified model.
        
        Args:
            model_id: HuggingFace model identifier
            quantization_bits: Optional quantization (4 or 8 bits)
        """
        self.model_id = model_id
        self.pipe = None
        self.quantization_bits = quantization_bits
        self.task_folder = None
        self.config : dict[str, Any] = {"": None}


    def load_model(self):
        """Load the model pipeline."""
        print(f"Loading model: {self.model_id}")
        
        # Authenticate with Hugging Face if needed
        if not authenticate_huggingface():
            raise ValueError("Failed to authenticate with Hugging Face")
        
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        if self.quantization_bits is not None:
            assert self.quantization_bits in (4, 8), "quantization_bits must be 4 or 8"
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=(self.quantization_bits==4), 
                load_in_8bit=(self.quantization_bits==8)
            )
        
        # Create pipeline for batched inference
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs=model_kwargs
        )
        print("Model loaded successfully!")


    def load_config(self, task_folder: Path):
        """
        Load configuration from JSON file in the task folder.
        
        Args:
            task_folder: Path to folder containing configuration and data files
        """
        self.task_folder = task_folder
        
        # Extract task name from folder path
        folder_name = os.path.basename(task_folder.name.rstrip('/'))
        
        # Try folder name with task_ prefix removed
        if folder_name.startswith('task_'):
            task_name = folder_name[5:]  # Remove "task_" prefix
            config_file = task_folder / f"{task_name}.json"
        else:
            task_name = folder_name
            config_file = task_folder / f"{task_name}.json"

        # If the expected config file doesn't exist, check for other JSON files
        if not config_file.exists():
            # Find all JSON files in the folder
            json_files = [f for f in task_folder.glob('*.json')]
            if len(json_files) == 0:
                raise FileNotFoundError(f"No JSON configuration files found in {task_folder}")
            elif len(json_files) == 1:
                # Single JSON file but doesn't match expected name
                actual_file = json_files[0]
                expected_name = f"{task_name}.json"
                raise FileNotFoundError(
                    f"Configuration file not found: {config_file}\n"
                    f"Found single JSON file '{actual_file}' but expected '{expected_name}'.\n"
                    f"Consider renaming '{actual_file}' to '{expected_name}'"
                )
            else:
                # Multiple JSON files - ambiguous
                raise FileNotFoundError(
                    f"Configuration file not found: {config_file}\n"
                    f"Multiple JSON files found: {json_files}\n"
                    f"Please ensure only one JSON file exists or rename it to match the expected name: {task_name}.json"
                )
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"Loaded configuration from {config_file}")
        print(f"Found {len(self.config)} tasks: {list(self.config.keys())}")


    def load_data_file(self, filename: str) -> pd.DataFrame:
        """
        Load data from a text file in the task folder.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            List of strings, one per line
        """
        if self.task_folder is None:
            raise ValueError("Task folder not set. Call load_config() first.")
        
        file_path = os.path.join(self.task_folder, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        df = pd.DataFrame(lines[1:], columns=lines[:1])
        
        return df


    def run_batch_inference(
        self, 
        messages: list[Any], 
        task_config: dict[str, Any]
    ) -> list[str]:
        """
        Run inference in batches using task configuration.

        Args:
            messages: List of message lists for batch inference
            task_config: Configuration for the current task

        Returns:
            List of responses for each message
        """
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        responses = []
        batch_size = task_config.get("batch_size", 8)
        max_new_tokens = task_config.get("max_new_tokens", 200)
        do_sample = task_config.get("do_sample", False)
        return_full_text = task_config.get("return_full_text", False)
        thinking_mode = task_config.get("thinking_mode", False)
        
        total_batches = (len(messages) + batch_size - 1) // batch_size 
        
        print(f"Processing {len(messages)} items in {total_batches} batches of size {batch_size}")
        
        for chunk in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_size * chunk
            end_idx = min(batch_size * (chunk + 1), len(messages))
            prompts = messages[start_idx:end_idx]
            
            if not prompts:  # Skip empty batches
                continue
                
            outputs = self.pipe(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                batch_size=len(prompts),
                return_full_text=return_full_text
            )
            
            for output in outputs:
                response = output[0]['generated_text']
                
                if thinking_mode:                    
                    if "<unused95>" in response:
                        _, response = response.split("<unused95>")
                        responses.append(response.strip().strip('\n'))
                    else:
                        # Sometimes, chain of thought does not terminate with <unused95>
                        response = response.strip().strip('\n')
                        
                        # heuristic 1: take the string after the last "Final Answer: ", if this substring is found in the response
                        final_answer_match = re.search(r'Final Answer:\s*(.*)', response)
                        if final_answer_match:
                            response = final_answer_match.group(1).strip()
                        
                        responses.append(response)
                else:
                    responses.append(response.strip().strip('\n'))
    
        return responses


    def save_output(self, data: pd.DataFrame, fp: str | Path):
        """
        Save output to a file in the task folder.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        data.to_csv(fp, index=False)
        print(f"Saved output to {fp}")


    def run_task(self, task_name: str, input_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Run a single task from the configuration.
        
        Args:
            task_name: Name of the task to run
            input_data: Optional input DataFrame (if not provided, loads from file)
            
        Returns:
            DataFrame with task results
        """
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        
        if task_name not in self.config:
            raise ValueError(f"Task '{task_name}' not found in configuration")
        
        task_config = self.config[task_name]
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print(f"DESCRIPTION: {task_config['description']}")
        print(f"{'='*60}")
        
        # Load input data
        if input_data is None:
            # Load from file specified in config
            data_file = task_config.get("data_file")
            if not data_file:
                raise ValueError(f"No data_file specified in task config")
            
            input_data = self.load_data_file(data_file)
        
        # Prepare each data entry for prompt formatting
        messages = []
        for _, row in input_data.iterrows():
            # add classification classes to prompt data
            prompt_data = {
                'classes': task_config['classes'],
            }
            # Add data fields specified in task config
            for field in task_config.get("data_fields", []):
                if field in row:
                    prompt_data[field] = row[field]
                else:
                    raise ValueError(f"Field '{field}' not found in input data")
            
            message = create_prompt(task_config, prompt_data)
            messages.append(message)
        
        # Run inference
        responses = self.run_batch_inference(messages, task_config)
        
        # Create output DataFrame
        result_df = input_data.copy()
        result_df[f'{task_name.upper()}_RESULT'] = responses
        
        return result_df


    def run_pipeline(self, fp: str | Path) -> dict[str, pd.DataFrame]:
        """
        Run the complete pipeline based on configuration.
        
        Args:
            task_folder: Path to folder containing configuration and data files
            
        Returns:
            Dictionary mapping task names to their result DataFrames
        """
        if isinstance(fp, str):
            task_folder = Path(fp)
        elif isinstance(fp, Path):
            task_folder = fp
        else:
            raise ValueError("task_folder must be a string or Path object")
        
        output_folder = Path(task_folder) / "out"
        output_folder.mkdir(exist_ok=True)

        # Load configuration
        self.load_config(task_folder)
        
        # Load model if not already loaded
        if self.pipe is None:
            self.load_model()
        
        current_data = None
        results = {}
        # Run tasks sequentially
        for i, task_name in enumerate(self.config.keys()):            
            # Run the task
            result_df = self.run_task(task_name, current_data)
                        
            # Save intermediate output
            output_filename = f"{i}_{task_name}_output.csv"
            self.save_output(result_df, output_folder / output_filename)

            results[task_name] = result_df
            current_data = result_df
            # Reset dynamo compilation between tasks
            torch._dynamo.reset()
            print(f"Completed task: {task_name}")
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED")
        print(f"{'='*60}")
        
        return results
        
        
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, 'pipe') and self.pipe is not None:
            del self.pipe
            self.pipe = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()