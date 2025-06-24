#!/usr/bin/env python3
"""
Quick test script to run classification on a small subset of procedures.
"""
import os
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')

from src.procedure_classification import ProcedureClassifier
import pandas as pd
import time
import matplotlib.pyplot as plt

def quick_test_with_validation(batch_size=None):
    """Run quick test with validation step."""
        
    # Initialize and load classifier
    if batch_size is not None:
        classifier = ProcedureClassifier(batch_size=batch_size)
    else:
        classifier = ProcedureClassifier()
    classifier.load_model()
    
    # Time the classification
    start_time = time.time()
    df = classifier.classify_procedures_from_file('procedure_test.txt')
    end_time = time.time()
    
    classification_time = end_time - start_time
    
    # Create dataframe and display initial results
    print("\n" + "="*80)
    print("INITIAL CLASSIFICATION RESULTS:")
    print("="*80)
    print(df.to_string(index=False))
    
    # Show category distribution
    print(f"\nCategory Distribution:")
    print(df['purpose'].value_counts())
    
    # Validate classifications
    print("\n" + "="*80)
    print("STARTING VALIDATION:")
    print("="*80)
    
    validation_df = classifier.validate_classifications(df, examples_per_category=3)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS:")
    print("="*80)
    print(validation_df[['procedure_name', 'purpose', 'validated_purpose', 'changed']].to_string(index=False))
    
    # Save results
    df.to_csv('quick_test_initial.csv', index=False)
    validation_df.to_csv('quick_test_validation.csv', index=False)
    print(f"\nInitial results saved to quick_test_initial.csv")
    print(f"Validation results saved to quick_test_validation.csv")
    
    return validation_df, classification_time

def batch_size_optimization_test():
    """Test different batch sizes and measure performance."""
    
    batch_sizes = [8, 16, 32, 64]
    results = []
    failed_batch_sizes = []
    
    print("Starting batch size optimization test...")
    print("="*80)
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch_size = {batch_size}")
        print("-" * 40)
        
        try:
            validation_df, classification_time = quick_test_with_validation(batch_size)
            num_procedures = len(validation_df)
            per_procedure_time = classification_time / num_procedures
            
            results.append({
                'batch_size': batch_size,
                'total_time': classification_time,
                'num_procedures': num_procedures,
                'per_procedure_time': per_procedure_time
            })
            
            print(f"Total time: {classification_time:.3f}s")
            print(f"Per-procedure time: {per_procedure_time:.3f}s")
            
        except Exception as e:
            print(f"Error with batch_size {batch_size}: {e}")
            failed_batch_sizes.append(batch_size)
            continue
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    results_df.to_csv('batch_size_results.csv', index=False)
    print(f"\nBatch size results saved to batch_size_results.csv")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['batch_size'], results_df['per_procedure_time'], 'bo-', linewidth=2, markersize=8)
    
    # Annotate failed runs with red asterisks
    if failed_batch_sizes:
        for failed_batch in failed_batch_sizes:
            plt.plot(failed_batch, 0, 'r*', markersize=20, label='Failed' if failed_batch == failed_batch_sizes[0] else "")
        plt.legend()
    
    plt.xlabel('Batch Size')
    plt.ylabel('Per-Procedure Inference Time (seconds)')
    plt.title('Per-Procedure Inference Time vs Batch Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Add value labels on points
    for i, row in results_df.iterrows():
        plt.annotate(f'{row["per_procedure_time"]:.3f}s', 
                    (row['batch_size'], row['per_procedure_time']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('batch_size_optim.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as batch_size_optim.png")
    
    # Print summary
    if not results_df.empty:
        best_batch_size = results_df.loc[results_df['per_procedure_time'].idxmin(), 'batch_size']
        best_time = results_df['per_procedure_time'].min()
        print(f"\nBest performance: batch_size={best_batch_size} with {best_time:.3f}s per procedure")
    
    if failed_batch_sizes:
        print(f"Failed batch sizes: {failed_batch_sizes}")
    
    return results_df

if __name__ == "__main__":
    print("Batch size optimization test")
    results = batch_size_optimization_test()
