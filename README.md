# Rx Copilot - Drug Indications from Labels

## Objective

Build a prototype software application that extracts indications from drug labels and maps
them to a standardized medical vocabulary, such as ICD-10 codes.

## Background

There is currently no definitive structured source that maps a given medical indication to all approved drugs that treat it. Drug labels, available freely via sources like DailyMed, contain this information, but it exists in unstructured text.

A key challenge is to:

* Extract the relevant text about indications from drug labels.
* Map these indications to a structured coding system (preferably ICD-10).
* Provide a structured and queryable dataset of drug-indication mappings.

This data could be used for:

* Search faceting in clinical software.
* Suggesting ICD-10 codes for use in prior authorization forms.
* Enabling more intelligent prescribing workflows.

## Requirements

* **Data Extraction:** Scrape or parse publicly available drug label data from DailyMed.
* **Indication Identification:** Extract sections of text that describe indications.
* **ICD-10 Mapping:** Attempt to map extracted indications to ICD-10 codes using an open-source ICD-10 dataset.
* **Prototype Implementation:** Build a minimal working prototype that takes a drug label and returns structured data mapping drugs to indications.
* **Documentation:** Provide clear instructions on:
    * How to set up and run the application.
    * Expected output.
    * Any limitations or challenges faced.
* **Edge Case Handling:** Consider nuances such as:
    * Synonyms (e.g., "Hypertension" vs. "High Blood Pressure").
    * Multi-indication drugs.
    * Rare conditions that may not map neatly to ICD-10.

## Deliverables

* A GitHub repository containing:
    * Source code for the prototype.
    * A `README.md` with:
        * Setup instructions.
        * Description of how the system works.
        * Sample output for a known drug.
        * Notes on how this could scale, potential improvements, and key challenges in productionizing this approach.

## Approach

This exercise involves several sequential problems. While a complete end-to-end solution is ideal, candidates should be prepared to discuss their methodology for addressing each step. It is understood that time constraints may prevent a fully coded solution for every aspect, but a clear plan for tackling each challenge is crucial.

Candidates should consider the following steps and be prepared to discuss their approach:

1.  ~~**Data Acquisition:** How to effectively retrieve drug label data from DailyMed or similar sources.~~
2.  **Text Extraction:** How to isolate and extract the sections of text that describe drug indications.
3.  **Indication Normalization:** How to standardize and clean the extracted indication text (e.g., handling synonyms, variations in phrasing).
4.  **ICD-10 Mapping:** How to link the normalized indications to corresponding ICD-10 codes.
5.  ~~**Data Structuring:** How to organize the extracted and mapped data into a structured format for querying and use.~~
6.  ~~**Prototype Development:** How to create a working prototype that demonstrates the core functionality of the system.~~

## Bonus Points

* Well-documented and modular code.
~~* A simple REST API that allows querying drug-indication mappings.~~
* Thoughts on scalability and production deployment considerations.
* Consideration of integrating AI/ML to improve text classification accuracy.


---

## Execution Strategy

### Goal

To develop a fast and accurate method to map drug indications from DailyMed datasets to ICD-10 codes, using FAISS for efficient fuzzy search. The end goal is to support natural language queries that return embedded drug indications and their mapped ICD-10 codes with a confidence interval.

In the spirit of prototyping and domain exploration, let's address the core value proposition of creating a simple yet reliably performant lightweight set of embeddings for DailyMed that can be replicated and tuned further for deeper investigation and productionization.

![methodology](./assets/method_tradeoffs.png)

#### Libraries
**Data Handling**: pandas, sqlite3

**NLP & Embeddings**: transformers, sentence-transformers, nltk, spacy

**Vector Search**: faiss

**ICD-10 Matching**: fuzzywuzzy, rapidfuzz

**Visualization**: matplotlib, seaborn

#### Step 1: Data Acquisition and Loading

1.	Download DailyMed Datasets:

    - Use [DailyMed](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) RSS and bulk download APIs.

    - Download the “Structured Product Labeling” (SPL) files, which contain XML files for all approved drugs.

2.	Extract Relevant Sections:

    - Focus on Indications and Usage and Dosage and Administration.

	- Parse XML files using libraries like lxml or xml.etree.ElementTree.

3.	Store Data:
	- Save structured data (drug name, indications text) into an SQLite database for quick access and indexing.

#### Step 2: Preprocessing and Embedding
1.	Text Preprocessing:
    - Remove stop words, special characters, and perform lemmatization.
	- Tokenization using spaCy or nltk.

2.	Generate Embeddings:
	- Use Hugging Face Transformers with pre-trained models (like sentence-transformers/all-MiniLM-L6-v2) to create sentence embeddings.

    - Store embeddings as vectors in a FAISS index.

#### Step 3: ICD-10 Code Mapping
1.	Load ICD-10 Codes:
    - Download a structured ICD-10 dataset (CSV or TSV).
	- Embed ICD-10 description texts using the same model.
	- Store the ICD-10 embeddings in a separate FAISS index.

2.	Fuzzy Matching with FAISS:
	- When querying with a new drug label or symptom, compute the embedding.
	- Use FAISS nearest neighbor search to find the closest matches in the ICD-10 vector space.
	- Compute a confidence score based on cosine similarity.

#### Step 4: Building the Query Pipeline
1.	Natural Language Query Handling:
	- Accept a text query from the user (e.g., a disease or indication).
	- Generate an embedding for the query using the same transformer model.

2.	Vector Search:
	- Query the FAISS index for the top K most similar ICD-10 embeddings.
	- Return the ICD-10 code, description, and confidence score.

#### Step 5: Output Format

Display the results in a pandas DataFrame:

![output](./output-format.png)

#### Step 6: Evaluation and Optimization
1.	Accuracy Check:
	- Manually verify a few common drugs to validate accuracy.
	- Adjust the threshold and model selection if necessary.

2.	Performance Optimization:
	- Experiment with different embedding models (e.g., sentence-transformers/all-distilroberta-v1).
	- Tune FAISS parameters for faster nearest neighbor retrieval.

#### Step 7: Productionization
1.	API with FastAPI:
	- Serve the model as a REST API for drug-indication mapping.
	- Endpoint: POST /map-indication with JSON payload containing indication text.

2.	Batch Processing:
	- Use Apache Spark or Dask for large-scale mapping.
	- Implement a data pipeline to regularly update the FAISS index with new SPL data.

#### Step 8: Human-in-the-Loop (HITL) Validation
1.	MD/Pharmacist Review:
	- Include a step for manual validation of top-ranked matches.
	- Collect feedback to iteratively improve accuracy and update ICD-10 mappings.

2.	Continuous Learning:
	- Implement feedback loops where human-validated mappings update the FAISS index and embedding models.