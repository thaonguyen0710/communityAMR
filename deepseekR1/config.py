# Configuration for the literature screening tool

import os
from dotenv import load_dotenv

# --- Specify the path to your .env file ---
# Use the absolute path you provided
dotenv_path = "screen.env"
# Check if the file exists before loading
if os.path.exists(dotenv_path):
    print(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}. Trying default location.")
    # Fallback to default behavior if specific path doesn't exist
    load_dotenv()

# Load environment variables from .env file
# load_dotenv() # Original line - replaced by specific path loading above

# Get API keys from environment variables
# Make sure DEEPSEEK_API_KEY matches the variable name in your screen.env file!
API_KEY = os.getenv("DEEPSEEK_API_KEY") 

# --- Screening Criteria --- 
# Define inclusion and exclusion criteria as a comprehensive prompt string
SCREENING_CRITERIA = """
You are helping screen studies for inclusion in a systematic review about antimicrobial resistance (AMR) carriage in the general population. 

# INCLUSION CRITERIA:
1. POPULATION: General population sampled from community settings (households, schools, workplaces)
   - Includes all ages, both sexes, and pregnant women
   - Can include individuals with chronic conditions, current/recent infections, or recent antibiotic use IF they were recruited from community settings

2. OUTCOME: Reports prevalence of AMR among selected pathogens from WHO 2024 BPPL
   - Pathogens include: Acinetobacter baumannii, Pseudomonas aeruginosa, Enterobacterales, Staphylococcus aureus, Enterococcus faecium, Streptococcus pneumoniae, Haemophilus influenzae, Salmonella spp., Shigella spp., Group A Streptococci, Group B Streptococci
   - Excludes: Mycobacterium tuberculosis and Neisseria gonorrhoeae

3. STUDY TYPES: Observational studies (cross-sectional, cohort, case-control) or RCTs with baseline AMR data
   - Must be peer-reviewed original research articles

# EXCLUSION CRITERIA:
1. SPECIFIC POPULATIONS:
   - Neonates in NICU or special care nurseries
   - Individuals with known immunodeficiencies or on immunosuppressive therapy
   - Healthcare workers (doctors, nurses, clinical staff)
   - Residents of long-term care facilities (nursing homes)
   - International travelers 
   - Occupational groups with unique exposures (veterinarians, animal handlers, pharmaceutical workers, farmers)
   - Clinical patients (outpatients, those being admitted to hospitals, those with diagnosed infections seeking care)

2. STUDY TYPES:
   - Conference abstracts/proceedings/posters
   - Systematic reviews/meta-analyses/review articles
   - Case reports/case series/case studies
   - Editorials/commentaries/opinion pieces/policy papers
   - Studies conducted during epidemics/outbreaks
   - Studies not in English, French, Spanish, or Mandarin
   - Preprints and unpublished studies

# SPECIFIC CASES TO CLARIFY:
1. "Community-acquired infection" studies should be EXCLUDED if they recruit patients with diagnosed infections from clinical settings
2. Studies on "pre-admission screening" should be EXCLUDED as these are clinical populations
3. Studies from nursing homes/long-term care facilities should be EXCLUDED
4. Studies on farms collecting data on farmers should be EXCLUDED as farmers have unique occupational exposures

Based on the title and abstract, determine if this study meets inclusion criteria or should be excluded. Provide a YES or NO decision with brief reasoning.
"""

# --- DeepSeek API Configurations ---
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
# Specify the DeepSeek model to use (e.g., deepseek-chat or deepseek-reasoner)
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat") # Default to deepseek-chat

def get_api_key():
    """Returns the loaded API key."""
    if not API_KEY:
        print("Warning: DEEPSEEK_API_KEY not found. Please set it in the .env file.")
    return API_KEY

def get_screening_criteria() -> str:
    """Returns the defined screening criteria prompt string."""
    return SCREENING_CRITERIA

def get_deepseek_api_base_url():
    """Returns the DeepSeek API base URL."""
    return DEEPSEEK_API_BASE_URL

def get_deepseek_model_name():
    """Returns the DeepSeek model name."""
    print(f"Using DeepSeek model: {DEEPSEEK_MODEL_NAME}")
    return DEEPSEEK_MODEL_NAME 