"""
Quick test script to run classification on a small subset of procedures using the new pipeline.
"""
import os
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
from src.procedure_classification import ProcedureClassifier


def clean_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def quick_test_with_pipeline(task_folder: str | Path):
    """Run quick test using the new pipeline approach."""
    print("\n" + "="*80)
    print("TESTING NEW PIPELINE APPROACH")
    print("="*80)
    
    classifier = ProcedureClassifier()
    
    try:
        # Time the entire pipeline
        start_time = time.time()
        results = classifier.run_pipeline(task_folder)
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nPipeline completed in {total_time:.2f} seconds")
        
        # Show results summary
        for task_name, result_df in results.items():
            print(f"\n--- Task: {task_name} ---")
            print(f"Processed {len(result_df)} procedures")
            
            result_col = f"{task_name}_result"
            if result_col in result_df.columns:
                print(f"Results distribution:")
                print(result_df[result_col].value_counts())
        
        return results, total_time
        
    finally:
        classifier.cleanup()


def batch_size_optimization_test(fp: str):
    """Test different batch sizes and measure performance using the new pipeline."""
    task_folder = Path(fp)
    if not task_folder.exists():
        raise ValueError(f"Task folder '{task_folder}' does not exist. Please provide a valid path.")
    
    batch_sizes = [2, 4, 8, 16, 32]
    results = []
    failed_batch_sizes = []
    
    for batch_size in batch_sizes:        
        print(f"\nTesting batch_size = {batch_size}")
        print("-" * 40)
                
        try:
            # Modify the config to use the specific batch size
            config_file = task_folder / 'test.json'
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Update batch sizes in all tasks
            for task_config in config.values():
                task_config['batch_size'] = batch_size
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Run pipeline with modified config
            pipeline_results, total_time = quick_test_with_pipeline(task_folder)
            
            # Get number of procedures from the first task
            first_task_df = list(pipeline_results.values())[0]
            num_procedures = len(first_task_df)
            
            results.append({
                'batch_size': batch_size,
                'total_time': total_time,
                'per_procedure_time': total_time / num_procedures,
                'num_procedures': num_procedures
            })
            
            print(f"✓ Batch size {batch_size} completed successfully")
            
        except Exception as e:
            print(f"✗ Error with batch_size {batch_size}: {e}")
            failed_batch_sizes.append(batch_size)
            clean_gpu_memory()
            
    # Create DataFrame with results
    results_df = pd.DataFrame(results)
    
    save_folder = task_folder / 'out'
    if not save_folder.exists():
        save_folder.mkdir(parents=True)

    results_df.to_csv(save_folder / 'batch_size_results.csv', index=False)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(results_df['batch_size'], results_df['per_procedure_time'], 'bo-', linewidth=2, markersize=8, label='Per-Procedure Time')
    
    # Annotate failed runs with red asterisks
    if failed_batch_sizes:
        for failed_batch in failed_batch_sizes:
            plt.plot(failed_batch, 0, 'r*', markersize=20, 
                    label='Failed' if failed_batch == failed_batch_sizes[0] else "")
    
    plt.xlabel('Batch Size')
    plt.ylabel('Per-Procedure Time (seconds)')
    plt.title('Per-Procedure Inference Time vs Batch Size (New Pipeline)')
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.legend()
    
    # Add value labels on points
    for i, row in results_df.iterrows():
        plt.annotate(
            f'{row["per_procedure_time"]:.3f}s', 
            xy=(row['batch_size'], row['per_procedure_time']),
            textcoords="offset points", xytext=(0,10), ha='center'
        )
    
    plt.tight_layout()
    plt.savefig(save_folder / 'batch_size_optim.png', dpi=300, bbox_inches='tight')

    # Print summary
    if not results_df.empty:
        best_batch_size = results_df.loc[results_df['per_procedure_time'].idxmin(), 'batch_size']
        best_time = results_df['per_procedure_time'].min()
        print(f"\nBest performance: batch_size={best_batch_size} with {best_time:.3f}s per procedure")
    
    if failed_batch_sizes:
        print(f"Failed batch sizes: {failed_batch_sizes}")


if __name__ == "__main__":    
    print("\n" + "="*80)
    print("Running batch size optimization test...")
    batch_size_optimization_test("task_test")
