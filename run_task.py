import sys
from src.procedure_classification import ProcedureClassifier

def main(task_folder: str):
    """
    Main function to run the configurable classification pipeline.
    
    Args:
        task_folder: Path to folder containing configuration and data files
    """
    # Initialize classifier
    classifier = ProcedureClassifier()
    
    try:
        # Run the complete pipeline
        results = classifier.run_pipeline(task_folder)
        
        # Print summary of all results
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        
        for task_name, result_df in results.items():
            print(f"\nTask: {task_name}")
            print(f"Processed {len(result_df)} items")
            
            # Show distribution if it's a classification result
            result_col = f"{task_name}_result"
            if result_col in result_df.columns:
                print("Result distribution:")
                print(result_df[result_col].value_counts())
                
    finally:
        # Clean up resources
        classifier.cleanup()


if __name__ == "__main__":    
    if len(sys.argv) != 2:
        raise ValueError("Please specify the task folder as a command line argument")
    
    task_folder = sys.argv[1] 
    results = main(task_folder)