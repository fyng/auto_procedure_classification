import os
import random
import pandas as pd
from tqdm import tqdm
import getpass
import torch

os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
from transformers import pipeline, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
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


def create_classification_prompt(categories, procedure_description: str) -> list[dict[str, str]]:
    """
    Create a classification prompt for the given procedure description.
    
    Args:
        procedure_description: The medical procedure description to classify
        
    Returns:
        Messages list for the LLM
    """
    classification_instruction = (
        f"Classify the following surgical procedure into one of these purpose categories: {categories}."
        f"Procedure: {procedure_description}."
        f"Return only a single category name and nothing else. \n\n"
    )
    
    messages = [
        {
            "role": "system",
            "content": "SYSTEM INSTRUCTION: think silently if needed. You are a helpful medical assistant with expertise in surgical procedures.",
        },
        {
            "role": "user", 
            "content": classification_instruction
        }
    ]
    
    return messages

    
def create_validation_prompt(categories, procedure_description: str, initial_category: str, example_procedures: list[str]) -> list[dict]:
    """
    Create a validation prompt with examples from the initially assigned category.
    
    Args:
        procedure_description: The medical procedure description to validate
        initial_category: The initially assigned category
        example_procedures: List of example procedures from the same category
        
    Returns:
        Messages list for the LLM
    """
    examples_text = "\n".join([f"- {proc}" for proc in example_procedures])
    
    validation_instruction = (
        f"You are validating the classification of a surgical procedure. "
        f"The procedure was initially classified as '{initial_category}'. "
        f"Here are 5 example procedures that were also classified as '{initial_category}':\n\n"
        f"{examples_text}\n\n"
        f"Now, please re-classify the following procedure into one of these categories: "
        f"{categories}. "
        f"Consider whether it truly belongs with the examples shown above, or if it should be in a different category. "
        f"Answer with a single category name.\n\n"
        f"Procedure to validate: {procedure_description}"
    )
    
    messages = [
        {
            "role": "system",
            "content": "SYSTEM INSTRUCTION: think silently if needed. You are a helpful medical assistant with expertise in surgical procedures. You are careful and precise in your classifications."
        },
        {
            "role": "user", 
            "content": validation_instruction
        }
    ]
    
    return messages


class ProcedureClassifier:
    """
    A classifier for medical procedure descriptions using MedGemma.
    Classifies procedures into purpose categories.
    """
    
    def __init__(
        self, 
        model_id: str = "google/medgemma-27b-text-it", 
        categories: list[str] | None = None,
        batch_size: int = 16
    ):
        """
        Initialize the classifier with the specified model.
        
        Args:
            model_id: HuggingFace model identifier
            batch_size: Batch size for processing
        """
        self.model_id = model_id
        self.pipe = None
        self.categories = categories if categories else self.default_categories()
        self.batch_size = batch_size

    def default_categories(self) -> list[str]:
        """Return default categories for procedure classification."""
        return [
            'Diagnostics', 'Tumor removal', 'Hemostasis',
            'Access', 'Therapeutic Ablation/Embolization',
            'Reconstruction/Repair', 'Other'
        ]
        
    def load_model(self):
        """Load the model pipeline."""
        print(f"Loading model: {self.model_id}")
        
        # Authenticate with Hugging Face if needed
        if not authenticate_huggingface():
            raise ValueError("Failed to authenticate with Hugging Face")
        
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True) # 4-bit quantization
        )
        
        # Create pipeline for batched inference
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs = model_kwargs
        )
        self.pipe.model.generation_config.do_sample = False # greedy decoding
    
        print("Model loaded successfully!")
        
    def run_inference(self, message: list[dict[str, str]]) -> str:
        """
        Run single inference using the pipeline.
        """ 
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output = self.pipe(
            message,
            max_new_tokens=1500,
            do_sample=False,
            return_full_text=False
        )
        
        response = output[0]['generated_text']
        # Handle the thought/response split if it exists
        if "<unused95>" in response:
            thought, response = response.split("<unused95>")
        
        return response.strip().strip('\n')
    
    
    def run_batch_inference(self, dataset: KeyDataset) -> list[str]:
        """
        Run inference in batches for speedup using the pipeline.

        Args:
            messages: List of message lists for batch inference

        Returns:
            List of responses for each message
        """
        # Use pipeline for batched inference
        responses = []
        for output in tqdm(self.pipe(dataset, max_new_tokens=1500, do_sample=False, batch_size=self.batch_size, return_full_text=False), desc="Processing batches"):
            response = output[0]['generated_text']
            if "<unused95>" in response:
                thought, response = response.split("<unused95>")
            responses.append(response.strip().strip('\n'))

        return responses
    
    
    def classify_procedures_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Classify procedures from a text file using batched inference and Dataset.

        Args:
            file_path: Path to the text file containing procedure descriptions
            batch_size: Number of procedures to process in each batch

        Returns:
            DataFrame with classified procedures (columns: procedure_name, purpose)
        """
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Create a dataset of prompts from procedure descriptions
        with open(file_path, 'r') as f:
            procedures = [line.strip() for line in f if line.strip()]
        messages = [create_classification_prompt(self.categories, proc) for proc in procedures]        
        dataset = Dataset.from_dict({"messages": messages})

        # Use pipeline for batched inference
        responses = self.run_batch_inference(KeyDataset(dataset, 'messages'))
                    
        # Convert back to DataFrame
        results_df = pd.DataFrame({
            'procedure_name': procedures,
            'purpose': responses
        })

        return results_df
        
    
    def validate_classifications(self, initial_df: pd.DataFrame, examples_per_category: int = 5) -> pd.DataFrame:
        """
        Validate all classifications using batch processing with context examples.
        
        Args:
            initial_df: DataFrame with initial classifications (columns: procedure_name, purpose)
            examples_per_category: Number of example procedures to show for each category
            
        Returns:
            DataFrame with validated classifications
        """
        if self.pipe is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print("Preparing validation data...")
        
        # Group procedures by category to get examples
        category_examples = {}
        for category in self.categories:
            category_procedures = initial_df[initial_df['purpose'] == category]['procedure_name'].tolist()
            if len(category_procedures) >= examples_per_category:
                category_examples[category] = random.sample(category_procedures, examples_per_category)
            else:
                category_examples[category] = category_procedures
        
        # Prepare validation dataset
        messages = []
        for idx, row in initial_df.iterrows():
            procedure = str(row['procedure_name'])
            initial_category = str(row['purpose'])
            
            # Get examples for this category
            examples = category_examples.get(initial_category, [])
            # Remove the current procedure from examples if it's there
            examples_filtered = [ex for ex in examples if ex != procedure][:examples_per_category]
            
            message = create_validation_prompt(self.categories, procedure, initial_category, examples_filtered)
            messages.append(message)
        
        dataset = Dataset.from_dict({"messages": messages})
        validation_data = self.run_batch_inference(KeyDataset(dataset, 'messages'))
        
        validation_df = initial_df.copy()
        validation_df['validated_purpose'] = validation_data
        validation_df['changed'] = validation_df['purpose'] != validation_df['validated_purpose']
        
        # Print summary statistics
        total_procedures = len(validation_df)
        changed_procedures = validation_df['changed'].sum()
        print(f"\nValidation Summary:")
        print(f"Total procedures: {total_procedures}")
        print(f"Classifications changed: {changed_procedures} ({changed_procedures/total_procedures*100:.1f}%)")
        print(f"Classifications unchanged: {total_procedures - changed_procedures} ({(total_procedures - changed_procedures)/total_procedures*100:.1f}%)")
        
        # Show category distribution changes
        print(f"\nCategory distribution comparison:")
        initial_counts = initial_df['purpose'].value_counts()
        validated_counts = validation_df['validated_purpose'].value_counts()
        
        comparison_df = pd.DataFrame({
            'Initial': initial_counts,
            'Validated': validated_counts,
            'Change': validated_counts - initial_counts
        }).fillna(0)
        print(comparison_df)
        
        return validation_df


def main():
    """Main function to run the classification pipeline."""
    # Initialize classifier
    classifier = ProcedureClassifier()
    
    # Load the model
    classifier.load_model()
    
    # Classify procedures from the input file using efficient method
    results_df = classifier.classify_procedures_from_file('procedure_name.txt')
    results_df.to_csv('procedure_classifications_initial.csv', index=False)
    print("Initial classifications saved to 'procedure_classifications_initial.csv'")
    
    # Display results
    print("\nInitial Classification Results:")
    print(results_df.head(10))
    
    # Show initial category distribution
    print("\nInitial Category Distribution:")
    print(results_df['purpose'].value_counts())
    
    # Validate classifications with context examples
    print("\n" + "="*60)
    print("STARTING VALIDATION PHASE")
    print("="*60)
    
    validation_df = classifier.validate_classifications(results_df, examples_per_category=5)
    
    # Save validation results
    validation_output_file = 'procedure_classifications_validated.csv'
    validation_df.to_csv(validation_output_file, index=False)
    print(f"\nValidation results saved to {validation_output_file}")
    
    # Create final results with validated classifications
    final_df = validation_df[['procedure_name', 'validated_purpose']].copy()
    final_df.columns = ['procedure_name', 'purpose']
    final_output_file = 'procedure_classifications_final.csv'
    final_df.to_csv(final_output_file, index=False)
    print(f"Final results saved to {final_output_file}")
    
    return validation_df

if __name__ == "__main__":
    results = main()