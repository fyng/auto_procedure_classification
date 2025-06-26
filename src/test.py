"""
Quick test script to run classification on a small subset of procedures.
"""
import os
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
from src.procedure_classification import ProcedureClassifier


def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    
def quick_test_with_validation(classifier: ProcedureClassifier):
    """Run quick test with validation step."""
    # Validate classifications
    print("\n" + "="*80)
    print("STAGE 1: CLASSIFICATION")
    print("="*80)
    
    # Time the classification
    start_time = time.time()
    df = classifier.classify_procedures_from_file('procedure_test.txt')
    end_time = time.time()
    classification_time = end_time - start_time
    torch._dynamo.reset()
    
    # Show category distribution
    print(f"\nCategory Distribution:")
    print(df['purpose'].value_counts())
    
    # Validate classifications
    print("\n" + "="*80)
    print("STAGE 2: VALIDATION")
    print("="*80)
    
    start_time = time.time()
    validation_df = classifier.validate_classifications(df)
    end_time = time.time()
    validation_time = end_time - start_time
    torch._dynamo.reset()
    
    # Save results
    df.to_csv('test_out/quick_test_initial.csv', index=False)
    validation_df.to_csv('test_out/quick_test_validation.csv', index=False)
    print(f"\nInitial results saved to quick_test_initial.csv")
    print(f"Validation results saved to quick_test_validation.csv")
    
    return validation_df, classification_time, validation_time


def batch_size_optimization_test():
    """Test different batch sizes and measure performance."""
    batch_sizes = [2, 4, 8, 16, 32]
    results = []
    failed_batch_sizes = []
    
    for batch_size in batch_sizes:
        clean_gpu_memory()
        
        classifier = ProcedureClassifier(batch_size=batch_size)
        classifier.load_model()
        
        print(f"\nTesting batch_size = {batch_size}")
        print("-" * 40)
        try:
            validation_df, class_time, val_time = quick_test_with_validation(classifier)
            num_procedures = len(validation_df)
            
            results.append({
                'batch_size': batch_size,
                'classification_time': class_time / num_procedures,
                'validation_time': val_time / num_procedures,
                'per_procedure_time': (class_time + val_time) / num_procedures
            })            
            del classifier
        except Exception as e:
            print(f"Error with batch_size {batch_size}: {e}")
            failed_batch_sizes.append(batch_size)   
            del classifier
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    results_df.to_csv('test_out/batch_size_results.csv', index=False)
    
        # Create plot with three lines
    plt.figure(figsize=(12, 8))
    # Plot three different metrics
    plt.plot(results_df['batch_size'], results_df['classification_time'], 'bo-',linewidth=2, markersize=8, label='Classification Time')
    plt.plot(results_df['batch_size'], results_df['validation_time'], 'ro-', linewidth=2, markersize=8, label='Validation Time')
    plt.plot(results_df['batch_size'], results_df['per_procedure_time'], 'go-', linewidth=2, markersize=8, label='Total Time')
    
    # Annotate failed runs with red asterisks
    if failed_batch_sizes:
        for failed_batch in failed_batch_sizes:
            plt.plot(failed_batch, 0, 'r*', markersize=20, label='Failed' if failed_batch == failed_batch_sizes[0] else "")
    
    plt.xlabel('Batch Size')
    plt.ylabel('Per-Procedure Time (seconds)')
    plt.title('Per-Procedure Inference Time vs Batch Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.legend()
    
    # Add value labels on points for total time
    for i, row in results_df.iterrows():
        plt.annotate(f'{row["per_procedure_time"]:.3f}s', 
                    (row['batch_size'], row['per_procedure_time']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('test_out/batch_size_optim.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved as batch_size_optim.png")
    
    # Print summary
    if not results_df.empty:
        best_batch_size = results_df.loc[results_df['per_procedure_time'].idxmin(), 'batch_size']
        best_time = results_df['per_procedure_time'].min()
        print(f"\nBest performance: batch_size={best_batch_size} with {best_time:.3f}s per procedure")
    
    if failed_batch_sizes:
        print(f"Failed batch sizes: {failed_batch_sizes}")
        

if __name__ == "__main__":
    if not os.path.exists('test_out'):
        os.makedirs('test_out')
        
    batch_size_optimization_test()
