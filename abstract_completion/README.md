# RIS Abstract Completion Tool

This tool automatically completes missing abstracts in RIS format reference data. It attempts to find and fill in missing abstracts using various methods, including CrossRef, PubMed, Semantic Scholar, and OpenAlex.

## Features

- Parses RIS format files and identifies references lacking abstracts.
- Attempts multiple methods to fetch abstracts:
  - Via DOI from CrossRef
  - Via title and authors from PubMed
  - Via title and authors from Semantic Scholar
  - Via title from OpenAlex
- Generates a detailed completion report, logging successes and failures.
- Outputs an updated RIS file, preserving the integrity of the original data.

## Dependencies Installation

Install the required dependencies using pip:

```bash
pip install requests habanero biopython
```

## Usage

Basic usage:

```bash
python abstract_completion.py your_references.ris
```

Specifying output file and report file:

```bash
python abstract_completion.py your_references.ris --output_file output.ris --report report.json
```

Using your email address (recommended for APIs) and setting sleep time:

```bash
python abstract_completion.py your_references.ris --email your.email@example.com --sleep 1.5
```

### Command-Line Arguments

- `input_file`: **Required**. Path to the input RIS file.
- `--output_file`: *Optional*. Path for the output RIS file. Defaults to `input_file_with_abstracts.ris`.
- `--report`: *Optional*. Path for the JSON completion report file. Defaults to `completion_report.json`.
- `--email`: *Optional*. Email address to use for API requests (e.g., for Entrez/PubMed, OpenAlex User-Agent). Defaults to `your.email@example.com`.
- `--sleep`: *Optional*. Time in seconds to pause between API requests. Defaults to `1.0`.

## Important Notes

1.  **Set Your Email**: Please set your email address either in the script (`Entrez.email = ...`) or using the `--email` command-line argument. This is required for PubMed API usage and recommended for others.
2.  **Network Connection**: The tool relies on online APIs. Ensure you have a stable internet connection.
3.  **API Rate Limits**: To avoid excessive API requests, the script pauses between requests (default 1 second, adjustable via `--sleep`). Some APIs (like Semantic Scholar) might still enforce rate limits. The script includes basic retry logic for Semantic Scholar's 429 errors.
4.  **Completion Success Rate**: The effectiveness depends on the availability of the references in the queried databases. Newer or less common publications might be harder to find abstracts for.
5.  **Large Files**: The script is designed to handle large RIS files by reading them line by line. 