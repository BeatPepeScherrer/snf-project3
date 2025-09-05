# NLP-based analysis of SDC credit proposals

## Overview

This repository contains the scripts used to (0) analyze a set of supplier code of conducts (CoCs) via langflow, (1) scrape stories and companies' responses to human rights violation allegations, and (2) feed the scraped json file into an LLM flow (directly coded) to answer several research questions.

## Scripts

### 0. `analyse_coc.py`

...

### 1. `bhrrc_scraper.py`

...

### 2. `analyse_bhrrc.py`

...


## Working with Langflow

Follow these steps to set up and work with Langflow:

1. Create an account at [DataStax](https://accounts.datastax.com/) to run Langflow directly in the cloud.
2. Create a new flow or import an existing one. 
3. 
4. 


## Setting up the Environment

1. Use a Python environment manager such as `conda` to manage Python versions:
   ```bash
   conda create -n myenv python=3.12.11
   conda activate myenv
   ```

2. Install the required dependencies via `pip` and `conda`:
   ```bash
    pip install -r requirements_scrape.txt # for bhrrc_scraper.py
    pip install -r requirements_llm.txt # for analyse_bhrrc.py and analyse_coc.py
    conda install -c conda-forge poppler # for pdf2image 
    conda install -c conda-forge tesseract pytesseract # also for reading of screenshot pdfs 
   ```

3. To use Langflow locally, refer to its documentation for launching it and adjust the URLs as needed for your environment. Ensure any custom configurations, such as API keys and endpoint URLs, are properly updated in the Langflow setup.
