#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
import requests
from habanero import Crossref
from Bio import Entrez
import random # For adding jitter

# Set user email for PubMed queries
Entrez.email = "your.email@example.com"  # Please change to your email

class AbstractFinder:
    def __init__(self, sleep_time=1):
        self.sleep_time = sleep_time
        self.crossref = Crossref(mailto=Entrez.email)
    
    def find_by_doi(self, doi):
        """Find abstract by DOI"""
        try:
            print(f"Attempting to fetch abstract for DOI: {doi} via CrossRef")
            time.sleep(self.sleep_time)  # Avoid high request frequency
            result = self.crossref.works(ids=doi)
            # Check result structure
            if result and 'message' in result and 'abstract' in result['message']:
                 # Clean HTML tags from CrossRef abstract
                 abstract_html = result['message']['abstract']
                 # Simple HTML tag removal
                 clean_abstract = re.sub('<[^<]+?>', '', abstract_html).strip()
                 return clean_abstract
            return None
        except Exception as e:
            print(f"Error fetching abstract via DOI: {e}")
            return None
    
    def find_by_pubmed(self, title, authors=None):
        """Find abstract via PubMed"""
        try:
            print(f"Attempting to fetch abstract for title '{title}' via PubMed")
            time.sleep(self.sleep_time)
            search_query = title
            if authors and len(authors) > 0:
                # Use more precise author matching
                author_query = " AND ".join([f"{auth.split(',')[0]}[Author]" for auth in authors[:2]]) # Use max first two authors
                search_query = f"({title}[Title]) AND ({author_query})"
                
            handle = Entrez.esearch(db="pubmed", term=search_query, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            # If exact match fails, try searching by title only
            if record["Count"] == "0" and authors:
                 print(f"PubMed exact match failed, trying search with title '{title}' only")
                 time.sleep(self.sleep_time)
                 handle = Entrez.esearch(db="pubmed", term=title, retmax=1)
                 record = Entrez.read(handle)
                 handle.close()

            if record["Count"] == "0":
                return None
            
            pmid = record["IdList"][0]
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text") # Get plain text abstract
            abstract_text = handle.read()
            handle.close()

            # Extract abstract - PubMed returns relatively simple text format
            # Usually the abstract content follows title/author info, can be used directly
            # Remove potential empty lines and leading/trailing spaces
            abstract = "\n".join(line for line in abstract_text.splitlines() if line.strip() and not re.match(r'^[A-Z\s]+:', line)) # Remove lines like "Author information:"
            return abstract.strip() if abstract else None

        except Exception as e:
            print(f"Error fetching abstract via PubMed: {e}")
            return None
            
    def find_by_semantic_scholar(self, title, authors=None, max_retries=3, initial_backoff=2):
        """Find abstract via Semantic Scholar, with retry and exponential backoff logic"""
        retries = 0
        backoff_time = initial_backoff
        while retries <= max_retries:
            try:
                print(f"Attempting to fetch abstract for title '{title}' via Semantic Scholar (Attempt {retries+1}/{max_retries+1})")
                # Pause before each actual request
                time.sleep(self.sleep_time)

                query_parts = [title]
                if authors:
                    query_parts.extend(authors[:2])
                query = "+".join(part.replace(' ', '+') for part in query_parts)

                url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=title,abstract"
                # Add User-Agent and potentially API Key (if obtained)
                headers = {'User-Agent': f'AbstractCompletionBot/1.0 (mailto:{Entrez.email})'}
                # If you have a Semantic Scholar API Key, add it like this:
                # headers['x-api-key'] = 'YOUR_SEMANTIC_SCHOLAR_API_KEY'

                response = requests.get(url, headers=headers, timeout=20) # Increased timeout
                response.raise_for_status() # Check for HTTP errors (including 429)

                data = response.json()

                if 'data' in data and len(data['data']) > 0:
                    best_match = None
                    best_score = 0.7
                    for paper in data['data']:
                         if 'abstract' in paper and paper['abstract']:
                            s_title = paper.get('title', '')
                            similarity = self._calculate_similarity(title, s_title)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = paper['abstract']
                            elif similarity > 0.8 and not best_match:
                                best_match = paper['abstract']

                    if not best_match and data['data'][0].get('abstract'):
                        print("Semantic Scholar: No highly similar title found, returning abstract of the first result as fallback.")
                        # return data['data'][0]['abstract'] # Optional: Uncomment if fallback is desired
                        return None # Current strategy: require high similarity

                    return best_match # Success, return result

                return None # API call successful but no data found

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retries += 1
                    if retries <= max_retries:
                        # Exponential backoff + random jitter
                        wait_time = backoff_time + random.uniform(0, 1)
                        print(f"Semantic Scholar API rate limit (429). Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
                        backoff_time *= 2 # Exponentially increase wait time
                    else:
                        print(f"Semantic Scholar API rate limit (429), max retries reached.")
                        return None # Max retries reached, give up
                else:
                    # Other HTTP errors
                    print(f"HTTP error occurred while requesting Semantic Scholar API: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                print(f"Network error occurred while requesting Semantic Scholar API: {e}")
                # Could consider retrying network errors, but not doing so for now
                return None
            except Exception as e:
                print(f"Unexpected error fetching abstract via Semantic Scholar: {e}")
                return None # Unknown error, don't retry

        return None # If loop finishes without success

    def find_by_openalex(self, title, authors=None):
        """Find abstract via OpenAlex"""
        try:
            print(f"Attempting to fetch abstract for title '{title}' via OpenAlex")
            time.sleep(self.sleep_time)
            # OpenAlex search prefers cleaned title
            search_title = re.sub(r'[^\w\s]', '', title).lower() # Clean title for better matching
            url = f"https://api.openalex.org/works?search={requests.utils.quote(search_title)}&filter=has_abstract:true&select=id,title,abstract_inverted_index"
            
            # Add contact email (recommended)
            headers = {'User-Agent': f'AbstractCompletionBot/1.0 (mailto:{Entrez.email})'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                best_match_abstract = None
                best_score = 0.7 # Similarity threshold

                for paper in data['results']:
                    # OpenAlex returns inverted index, need to reconstruct abstract
                    if 'abstract_inverted_index' in paper and paper['abstract_inverted_index']:
                        abstract = self._reconstruct_abstract_from_inverted_index(paper['abstract_inverted_index'])
                        if abstract:
                            o_title = paper.get('title', '')
                            similarity = self._calculate_similarity(title, o_title)
                            
                            if similarity > best_score:
                                best_score = similarity
                                best_match_abstract = abstract
                            elif similarity > 0.8 and not best_match_abstract: # Fallback
                                 best_match_abstract = abstract

                # If no highly similar match, consider the first result
                if not best_match_abstract and data['results'][0].get('abstract_inverted_index'):
                     first_abstract = self._reconstruct_abstract_from_inverted_index(data['results'][0]['abstract_inverted_index'])
                     if first_abstract:
                          print("OpenAlex: No highly similar title found, returning abstract of the first result as fallback.")
                          # return first_abstract # Optional: Uncomment if fallback is desired
                          return None # Current strategy: require high similarity

                return best_match_abstract
            return None
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred while requesting OpenAlex API: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching abstract via OpenAlex: {e}")
            return None
            
    def _reconstruct_abstract_from_inverted_index(self, inverted_index):
        """Reconstruct abstract from OpenAlex's inverted index"""
        if not inverted_index:
            return None
        
        word_positions = {}
        max_pos = 0
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions[pos] = word
                if pos > max_pos:
                    max_pos = pos
        
        # Build abstract by position order
        abstract_words = [word_positions.get(i, '') for i in range(max_pos + 1)]
        return " ".join(filter(None, abstract_words)) # Filter empty strings and join

    def _calculate_similarity(self, str1, str2):
        """Calculate Jaccard similarity between two strings (based on word sets)"""
        if not str1 or not str2:
             return 0
        # Preprocess: lower, remove punctuation, tokenize
        s1_words = set(re.findall(r'\b\w+\b', str1.lower()))
        s2_words = set(re.findall(r'\b\w+\b', str2.lower()))
        
        if not s1_words or not s2_words:
            return 0
            
        intersection = len(s1_words.intersection(s2_words))
        union = len(s1_words.union(s2_words))
        return intersection / union if union > 0 else 0

def parse_ris(file_path):
    """Improved RIS parsing function, reads line by line for large files, handles tags and end markers more robustly"""
    references = []
    current_ref = {}
    current_tag = None
    last_line_tag = False # Flag if the previous line was a tag line

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                stripped_line = line.strip()

                # Standard end marker
                if stripped_line == "ER  -":
                    if current_ref:
                        references.append(current_ref)
                    current_ref = {}
                    current_tag = None
                    last_line_tag = False
                    continue

                # Handle potential non-standard ER markers or missing ER at EOF
                # (Logic might need refinement for very large files without ER)

                if not stripped_line: # Skip empty lines
                    last_line_tag = False
                    continue

                # Check if it's a new tag line (TY-PY, A1-A4, AU, etc.)
                match = re.match(r'^([A-Z][A-Z0-9])  - (.*)', line) # Match original line to preserve spacing
                if match:
                    current_tag = match.group(1)
                    content = match.group(2).strip() # Strip leading/trailing whitespace from content
                    last_line_tag = True

                    # Handle multi-value tags (e.g., AU)
                    if current_tag in ['AU', 'KW', 'A1', 'A2', 'A3', 'A4']: # Extendable list
                        if current_tag not in current_ref:
                            current_ref[current_tag] = []
                        current_ref[current_tag].append(content)
                    # Handle single-value tags (overwrite previous, last one wins)
                    else:
                         current_ref[current_tag] = content
                
                # Handle continuation lines (follow a tag line, don't start with tag format)
                elif current_tag and not last_line_tag and stripped_line: 
                    # Append continuation content to the last value (for multi-line AB, N1 etc.)
                    if current_tag in current_ref:
                         # If it's a list (like AU), shouldn't technically have continuations, but append to last element just in case
                         if isinstance(current_ref[current_tag], list):
                              if current_ref[current_tag]: # Ensure list is not empty
                                   current_ref[current_tag][-1] += " " + stripped_line
                         # If it's a string, append directly
                         else:
                              current_ref[current_tag] += " " + stripped_line
                    # If tag wasn't recorded (shouldn't happen), ignore this continuation
                    # else:
                    #      print(f"Warning: Orphan continuation line found at line {line_num}: {stripped_line}")


                # If it's not a tag line, nor a continuation (could be format error or unexpected line)
                elif not last_line_tag:
                      # print(f"Warning: Non-standard format line found at line {line_num}: {stripped_line}")
                      pass # Choose to ignore or log these lines
                
                # Update flag as current line wasn't a tag line
                if not match:
                    last_line_tag = False


        # Handle the last reference if file doesn't end with ER marker
        if current_ref:
            references.append(current_ref)
            
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return []
    except Exception as e:
        print(f"Unexpected error parsing RIS file: {e}")
        return [] # Return empty list on failure

    return references

def write_ris(references, output_file):
    """Write the list of references to an RIS file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, ref in enumerate(references):
                for tag, content in ref.items():
                    if isinstance(content, list):
                        for item in content:
                            f.write(f"{tag}  - {item}\n")
                    else:
                        # RIS standard often requires indenting multi-line fields like AB, but simplified here
                        f.write(f"{tag}  - {content}\n")
                f.write("ER  - \n")
                # Add a blank line between records for readability
                if i < len(references) - 1:
                    f.write("\n")
    except IOError as e:
        print(f"Error writing RIS file: {e}")
    except Exception as e:
        print(f"Unexpected error writing RIS file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Complete missing abstracts in RIS files')
    parser.add_argument('input_file', help='Input RIS file path')
    parser.add_argument('--output_file', help='Output RIS file path', default=None)
    parser.add_argument('--report', help='Path for the JSON completion report file', default='completion_report.json')
    parser.add_argument('--email', help='Email address for API requests', default="your.email@example.com") # Add email arg
    parser.add_argument('--sleep', type=float, help='Sleep time between API requests (seconds)', default=1.0) # Add sleep time arg

    args = parser.parse_args()
    
    # Update Entrez email
    Entrez.email = args.email
    print(f"Using email: {args.email} for API requests")

    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_with_abstracts.ris"
    
    print(f"Parsing RIS file: {args.input_file}")
    references = parse_ris(args.input_file)
    
    if not references:
         print("Failed to parse any references. Please check file format or path.")
         return # Exit if parsing failed

    print(f"Found {len(references)} references")
    
    # Pass email and sleep time to AbstractFinder
    abstract_finder = AbstractFinder(sleep_time=args.sleep) 
    missing_abstracts_count = 0
    completed_abstracts_count = 0
    failed_abstracts_list = []
    
    # Use enumerate to get index and reference object
    for i, ref in enumerate(references):
        # Check if abstract is missing (AB tag)
        # Also check if abstract content is empty or just whitespace
        has_abstract = 'AB' in ref and ref['AB'] and ref['AB'].strip()

        if not has_abstract:
            missing_abstracts_count += 1
            print(f"\nProcessing reference {i+1}/{len(references)} (Missing Abstract):")
            
            # Get reference info, provide defaults in case tags are missing
            title = ref.get('TI', ref.get('T1', '')) # Some RIS use T1 for title
            doi = ref.get('DO', ref.get('DI', ''))   # Some use DI for DOI
            authors = ref.get('AU', [])
            # Ensure authors is a list
            if authors and not isinstance(authors, list):
                authors = [authors]
            
            # Print key info, truncate long titles
            print(f"  Title: {title[:80] + '...' if len(title) > 80 else title}")
            print(f"  DOI: {doi if doi else 'N/A'}")
            print(f"  Authors: {', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else '') if authors else 'N/A'}")
            
            abstract = None
            
            # Strategy 1: Try DOI (CrossRef)
            if doi:
                abstract = abstract_finder.find_by_doi(doi)
                if abstract: print("  > Successfully found abstract via DOI (CrossRef)")

            # Strategy 2: Try PubMed (Title+Authors / Title only)
            if not abstract and title:
                abstract = abstract_finder.find_by_pubmed(title, authors)
                if abstract: print("  > Successfully found abstract via PubMed")

            # Strategy 3: Try Semantic Scholar (Title+Authors / Title only)
            if not abstract and title:
                abstract = abstract_finder.find_by_semantic_scholar(title, authors)
                if abstract: print("  > Successfully found abstract via Semantic Scholar")

            # Strategy 4: Try OpenAlex (Title)
            if not abstract and title:
                 abstract = abstract_finder.find_by_openalex(title, authors) # Pass authors for potential future use
                 if abstract: print("  > Successfully found abstract via OpenAlex")
            
            # Update reference object
            if abstract:
                # Clean up retrieved abstract, remove excess newlines/spaces
                cleaned_abstract = ' '.join(abstract.split())
                ref['AB'] = cleaned_abstract
                completed_abstracts_count += 1
                # print(f"Successfully found abstract!") # Moved success message to specific methods
            else:
                failed_abstracts_list.append({
                    'index': i+1, # Use 1-based index for report
                    'title': title,
                    'doi': doi,
                    'authors': authors
                })
                print(f"  > Failed to find abstract using all methods")
        else:
             # If reference already has an abstract, print a message or skip silently
             # print(f"Reference {i+1}/{len(references)} already has an abstract, skipping.")
             pass # Silently skip references with existing abstracts

    
    # Write updated RIS file
    print(f"\nWriting results to file: {args.output_file}")
    write_ris(references, args.output_file)
    
    # Generate report
    report = {
        'input_file': args.input_file,
        'output_file': args.output_file,
        'total_references': len(references),
        'initial_missing_abstracts': missing_abstracts_count, # Record initial count
        'completed_abstracts': completed_abstracts_count,
        'failed_abstracts_count': len(failed_abstracts_list),
        'failed_references': failed_abstracts_list # Include details of failed references
    }
    
    print(f"Generating report file: {args.report}")
    try:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except IOError as e:
         print(f"Error writing report file: {e}")
    except Exception as e:
         print(f"Unexpected error writing report file: {e}")

    
    print(f"\n--- Abstract Completion Finished ---")
    print(f"Total references: {len(references)}")
    print(f"Initially missing abstracts: {missing_abstracts_count}")
    print(f"Successfully completed: {completed_abstracts_count}")
    print(f"Finally failed to complete: {len(failed_abstracts_list)}")
    print(f"Completed file saved to: {args.output_file}")
    print(f"Completion report saved to: {args.report}")
    print("-------------------------------------")

if __name__ == "__main__":
    main() 