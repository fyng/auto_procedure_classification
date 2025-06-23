#!/usr/bin/env python3
"""
Quick test script to run classification on a small subset of procedures.
"""
import os
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HUB_CACHE'] = os.path.join(os.getcwd(), 'hf_cache')

from procedure_classification import ProcedureClassifier
import pandas as pd

def quick_test_with_validation():
    """Run quick test with validation step."""
        
    # Initialize and load classifier
    classifier = ProcedureClassifier()
    classifier.load_model()
    
    df = classifier.classify_procedures_from_file('procedure_test.txt')
    
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
    
    return validation_df

if __name__ == "__main__":
    print("Quick test with validation (10 procedures)")
    results = quick_test_with_validation()
