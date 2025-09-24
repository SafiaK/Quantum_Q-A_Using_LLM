#!/usr/bin/env python3
"""
Run pipeline on first 10 rows with increased timeout
"""

from run_pipeline import run_pipeline_on_first_n_rows, save_results

def main():
    """
    Run pipeline on first 10 rows
    """
    # Configuration
    filtered_file = "filtered_quantum_dataset.csv"
    examples_file = "few_shot_examples.json"
    num_rows = 10  # first 10 rows
    start_row = 0  # start from row 0
    
    try:
        print("Running pipeline on first 10 rows with 180s timeout...")
        print("="*60)
        
        # Run pipeline on first 10 rows
        results = run_pipeline_on_first_n_rows(
            filtered_file=filtered_file,
            examples_file=examples_file,
            num_rows=num_rows,
            start_row=start_row
        )
        
        # Save results
        output_file = f"generated_answers_first_10_rows.csv"
        save_results(results, output_file)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Show summary
        print(f"\nSUMMARY:")
        print(f"Total questions processed: {len(results)}")
        prompt1_success = sum(1 for r in results if r['prompt1_success'])
        prompt2_success = sum(1 for r in results if r['prompt2_success'])
        print(f"Prompt 1 success: {prompt1_success}/{len(results)} ({prompt1_success/len(results)*100:.1f}%)")
        print(f"Prompt 2 success: {prompt2_success}/{len(results)} ({prompt2_success/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
