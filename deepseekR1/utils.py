from typing import Optional, Dict
import requests
import json
import re
from config import get_api_key, get_deepseek_api_base_url, get_deepseek_model_name # Import config functions
import rispy
import pandas as pd
from typing import List

# --- Data Loading Functions ---
def load_literature_ris(filepath: str) -> Optional[pd.DataFrame]:
    """Loads literature data from a RIS file and returns a pandas DataFrame.

    Args:
        filepath: The path to the RIS file.

    Returns:
        A pandas DataFrame containing literature data (e.g., title, abstract),
        or None if the file cannot be read or is empty.
    """
    try:
        # Specify encoding explicitly, utf-8 is common but others might be needed
        with open(filepath, 'r', encoding='utf-8') as bibliography_file:
            entries: List[Dict] = list(rispy.load(bibliography_file))
        
        if not entries:
            print(f"Warning: No entries found in RIS file: {filepath}")
            return None

        # Extract relevant fields (adjust fields based on your RIS file structure)
        data_for_df = []
        for entry in entries:
            # RIS fields can vary; try common tags for title and abstract
            title = entry.get('title') or entry.get('primary_title') or entry.get('TI') or entry.get('T1')
            abstract = entry.get('abstract') or entry.get('AB') or entry.get('N2')
            
            # Add other fields you might need, e.g., authors, year
            authors = entry.get('authors') or entry.get('AU')
            year = entry.get('year') or entry.get('PY')

            data_for_df.append({
                'title': title,
                'abstract': abstract,
                'authors': authors, # Keep as list or join
                'year': year
                # Add more fields as needed
            })

        df = pd.DataFrame(data_for_df)
        print(f"Successfully loaded {len(df)} entries from {filepath}")
        return df

    except FileNotFoundError:
        print(f"Error: RIS file not found at {filepath}")
        return None
    except UnicodeDecodeError as e:
         print(f"Error decoding RIS file {filepath} with UTF-8: {e}")
         print("Try specifying a different encoding if needed (e.g., 'latin-1', 'iso-8859-1').")
         return None
    except Exception as e:
        print(f"Error reading or parsing RIS file {filepath}: {e}")
        return None

# --- Placeholder for CSV loading (if needed later) ---
def load_literature_csv(filepath: str) -> Optional[pd.DataFrame]:
    """Placeholder function to load literature data from a CSV file."""
    print("CSV loading function not fully implemented yet.")
    # Implementation would use pd.read_csv, handle columns, errors
    try:
        df = pd.read_csv(filepath)
        # Add validation for expected columns like 'Title', 'Abstract'
        if 'Title' not in df.columns or 'Abstract' not in df.columns:
            print(f"Error: CSV file {filepath} missing required columns 'Title' or 'Abstract'.")
            return None
        print(f"Successfully loaded {len(df)} entries from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return None

# --- Placeholder for BibTeX loading (if needed later) ---
def load_literature_bibtex(filepath: str) -> Optional[pd.DataFrame]:
    """Placeholder function to load literature data from a BibTeX file."""
    print("BibTeX loading function not fully implemented yet.")
    # Implementation would use bibtexparser
    return None

# --- LLM Prompt Construction ---
def construct_llm_prompt(abstract: Optional[str], criteria_prompt: str) -> Optional[str]:
    """Constructs the prompt for the LLM based on the abstract and a detailed criteria prompt string.

    Args:
        abstract: The abstract text of the literature.
        criteria_prompt: A detailed string containing the screening instructions and criteria.

    Returns:
        A formatted prompt string, or None if the abstract is missing.
    """
    if not abstract or not isinstance(abstract, str) or abstract.strip() == "":
        # Handle cases where abstract is missing or not a valid string
        return None

    # The criteria_prompt already contains screening instructions.
    # We need to ensure the final part asks for INCLUDE/EXCLUDE/UNCERTAIN and the correct format.
    # Remove the last part of criteria_prompt asking for YES/NO if it exists
    criteria_lines = criteria_prompt.strip().split('\n')
    if criteria_lines[-1].startswith("Based on the title and abstract"): 
        criteria_prompt_base = "\n".join(criteria_lines[:-1]).strip()
    else:
        criteria_prompt_base = criteria_prompt.strip()


    prompt = f"""{criteria_prompt_base}

# Screening Task:
Based *only* on the abstract provided below, classify the study using ONE of the following labels: INCLUDE, EXCLUDE, UNCERTAIN.
Then, provide a brief (1-2 sentence) justification for your decision.

# Study Abstract:
---
{abstract}
---

# Your Classification:
Format your response EXACTLY as follows (LABEL and Justification on separate lines):
LABEL: [Your Decision - INCLUDE, EXCLUDE, or UNCERTAIN]
Justification: [Your Brief Justification]"""
    
    return prompt


# --- API Calling Function (Adapted for DeepSeek API - OpenAI Compatible) ---
def call_screening_api(prompt_text: str) -> Optional[Dict[str, str]]:
    """Calls the DeepSeek API (OpenAI compatible) with the provided prompt.

    Args:
        prompt_text: The formatted prompt string.

    Returns:
        A dictionary containing 'label' and 'justification', or None if the call fails.
    """
    api_key = get_api_key()
    base_url = get_deepseek_api_base_url()
    model_name = get_deepseek_model_name()

    if not api_key:
        print("Error: DEEPSEEK_API_KEY is missing. Cannot call API.")
        return None
    if not model_name:
        print("Error: DeepSeek model name is not configured. Cannot call API.")
        return None
    if not base_url:
        print("Error: DeepSeek API base URL is not configured. Cannot call API.")
        return None

    # Construct the API endpoint URL
    api_endpoint = f"{base_url.rstrip('/')}/chat/completions" # Standard OpenAI endpoint
    
    headers = {
        "Authorization": f"Bearer {api_key}", # Use Bearer token auth
        "Content-Type": "application/json",
    }

    # --- OpenAI Compatible Payload --- 
    # Add a system message for better context, similar to DeepSeek docs example
    messages = [
        {"role": "system", "content": "You are an AI assistant performing literature screening for a research study. Follow the user's instructions precisely and provide output in the specified format."},
        {"role": "user", "content": prompt_text}
    ]
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.5, # Adjust as needed
        "max_tokens": 150,  # Adjust as needed for label + justification
        # "top_p": ...,
        "stream": False # We want the full response at once
        # Add other parameters supported by DeepSeek if needed
    }
    # --- End of payload section ---

    try:
        print(f"   - Sending request to DeepSeek model: {model_name} at {api_endpoint}") 
        response = requests.post(api_endpoint, headers=headers, json=data, timeout=90) 
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_json = response.json()
        # print(f"   - Raw API Response: {json.dumps(response_json, indent=2)}") # Uncomment for deep debugging

        # --- OpenAI Compatible Response Parsing --- 
        # Extract the generated text from the response
        if 'choices' in response_json and len(response_json['choices']) > 0:
            choice = response_json['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                message_content = choice['message'].get('content', '')
                
                # Check finish reason if needed (OpenAI format)
                finish_reason = choice.get('finish_reason')
                if finish_reason and finish_reason != "stop":
                     print(f"   - Warning: API call finished with reason: {finish_reason}")
                     # Handle length, content filter etc. based on DeepSeek documentation if needed
                     # For now, just proceed but be aware.

                print(f"   - Received content: '{message_content[:100]}...'") 
                
                # Parse the LABEL and Justification from the message content
                # Reverted regex to match INCLUDE, EXCLUDE, UNCERTAIN for the label
                label_match = re.search(r"^LABEL:\s*(INCLUDE|EXCLUDE|UNCERTAIN)", message_content, re.IGNORECASE | re.MULTILINE)
                justification_match = re.search(r"^Justification:\s*(.*)", message_content, re.IGNORECASE | re.MULTILINE)
                
                label = label_match.group(1).upper() if label_match else "PARSE_ERROR"
                justification = justification_match.group(1).strip() if justification_match else "Could not parse justification from API response."
                
                if label == "PARSE_ERROR":
                     print(f"   - Error: Could not parse LABEL from response: {message_content}")
                
                return {"label": label, "justification": justification}
            else:
                print(f"Error: Malformed response - Missing 'message' or 'content' in choice: {choice}")
                return None
        else:
            # Check for potential top-level error message (format might vary)
            error_message = response_json.get('error', {}).get('message', 'Unknown error format')
            print(f"Error: Unexpected API response format - Missing 'choices' or potentially an API error: {error_message}")
            return None
        # --- End of response parsing section ---

    except requests.exceptions.Timeout:
        print(f"Error calling DeepSeek API: Request timed out after 90 seconds.")
        return None
    except requests.exceptions.RequestException as e:
        error_details = ""
        if e.response is not None:
            try:
                error_json = e.response.json()
                error_details = f" - Status Code: {e.response.status_code}, Response: {json.dumps(error_json)}"
            except json.JSONDecodeError:
                error_details = f" - Status Code: {e.response.status_code}, Response: {e.response.text}"
        print(f"Error calling DeepSeek API: {e}{error_details}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode API response JSON: {response.text}")
        return None
    except Exception as e:
        import traceback 
        print(f"An unexpected error occurred during API call or parsing: {e}")
        traceback.print_exc() 
        return None
