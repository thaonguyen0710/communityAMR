# Main script for the literature screening tool

import pandas as pd
import time
import argparse # For command-line arguments

# Import utility functions and configuration getters
from utils_gemini import load_literature_ris, construct_llm_prompt, call_screening_api
from config_gemini import get_screening_criteria # We get API key/endpoint inside call_screening_api


def main(input_filepath: str, output_filepath: str, delay: float = 1.0):
    """Main function to run the literature screening process."""
    print("--- Starting Literature Screening Process ---")

    # 1. Load Configuration (Criteria)
    print("Loading screening criteria...")
    criteria = get_screening_criteria()
    # API key/endpoint are loaded within call_screening_api

    # 2. Load Literature Data
    print(f"Loading literature data from: {input_filepath}")
    df = load_literature_ris(input_filepath)

    if df is None:
        print("Failed to load literature data. Exiting.")
        return
    
    if 'abstract' not in df.columns:
        print("Error: 'abstract' column not found in the loaded data. Cannot proceed.")
        # You might want to try other abstract column names here if needed
        return

    print(f"Loaded {len(df)} entries.")

    # 3. Initialize columns for results
    df['AI_Decision'] = "PENDING"
    df['AI_Reasoning'] = ""

    # 4. Iterate, Prompt, Call API, Store Results
    print("Starting screening...")
    total_entries = len(df)
    screened_count = 0
    error_count = 0

    for index, row in df.iterrows():
        abstract = row.get('abstract') # Use .get() for safety
        entry_identifier = row.get('title', f"Entry {index + 1}") # Use title or index for logging

        print(f"\nProcessing entry {index + 1}/{total_entries}: '{entry_identifier[:50]}...'")

        if pd.isna(abstract) or not abstract or not isinstance(abstract, str) or abstract.strip() == "":
            print(" - Abstract is missing or empty. Skipping.")
            df.loc[index, 'AI_Decision'] = "NO_ABSTRACT"
            continue

        # 4a. Construct Prompt
        prompt = construct_llm_prompt(abstract, criteria)
        if prompt is None: # Should not happen if abstract check passed, but for safety
             print(" - Failed to construct prompt (unexpected). Skipping.")
             df.loc[index, 'AI_Decision'] = "PROMPT_ERROR"
             continue
        
        # print(f"   - Generated Prompt (first 100 chars): {prompt[:100]}...") # Uncomment for debugging

        # 4b. Call API
        print(" - Calling LLM API...")
        api_result = call_screening_api(prompt)

        # 4c. Store Result
        if api_result and isinstance(api_result, dict):
            label = api_result.get('label', 'API_ERROR')
            justification = api_result.get('justification', 'No justification returned.')
            df.loc[index, 'AI_Decision'] = label
            df.loc[index, 'AI_Reasoning'] = justification
            print(f"   - API Result: {label}")
            screened_count += 1
        else:
            print("   - API call failed or returned invalid data.")
            df.loc[index, 'AI_Decision'] = "API_ERROR"
            error_count += 1

        # 5. Delay between calls
        if delay > 0:
            # print(f"   - Waiting for {delay} seconds...") # Uncomment for debugging
            time.sleep(delay)

    print("\n--- Screening Finished ---")
    print(f"Successfully screened: {screened_count}")
    print(f"Entries skipped (no abstract): {len(df[df['AI_Decision'] == 'NO_ABSTRACT'])}")
    print(f"API/Processing errors: {error_count}")

    # 6. Save Output (moved to step 9, called after main)
    # print(f"\nSaving results to: {output_filepath}")
    # try:
    #     df.to_csv(output_filepath, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    #     print("Results saved successfully.")
    # except Exception as e:
    #     print(f"Error saving results to {output_filepath}: {e}")

    return df # Return the dataframe for potential further use or saving


if __name__ == "__main__":
    # Set up argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Screen literature using an LLM API.")
    parser.add_argument("input_file", help="Path to the input RIS literature file.")
    parser.add_argument("-o", "--output_file", default="output_screened_literature.csv", 
                        help="Path to save the output CSV file (default: output_screened_literature.csv)")
    parser.add_argument("-d", "--delay", type=float, default=1.0,
                        help="Delay in seconds between API calls (default: 1.0)")
    
    args = parser.parse_args()

    # Run the main screening process
    results_df = main(args.input_file, args.output_file, args.delay)

    # Step 9: Implement Output Saving
    if results_df is not None:
        print(f"\nSaving results to: {args.output_file}")
        try:
            results_df.to_csv(args.output_file, index=False, encoding='utf-8-sig') # Use utf-8-sig for better Excel compatibility
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to {args.output_file}: {e}")
    else:
        print("Skipping saving results as the process did not complete successfully.")

    # Reminder for human verification
    print("\nIMPORTANT: AI screening results require careful manual verification.") 