# textbook-qa
Building a Question Answering system with [ALBERT](https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html) on SQuAD dataset for large documents such as textbooks.

## Overview
Modern information retrieval techniques are successful in retrieving information from smaller documents. However, when it comes to larger documents, current options fall short. This repository attempts to solve the problem of performing Question Answering on large documents. This requires a two part approach. In one part, ALBERT is trained on the SQuAD QA dataset. In the other, we fragment a textbook into multiple sections using a rule based approach. We can then compare user question embeddings to the embeddings of the sections to find the most relevant section(s). 

Both parts come together when relevant sections along with the user question are passed to ALBERT to produce the predicted answer:

![Diagram](resources/diagram.png)

To follow along with any section(s) below, clone this reposiroty:

```git clone https://github.com/devkosal/albert-qa/ | cd albert-qa```

and install requirements:

```pip install -r requirements.txt```


## Quick Demo 

This app uses Pyviz's Panel to serve the end application. Run the following command to serve the app locally:

```panel serve --show "TextbookQA Sample App.ipynb"```

## Building Embeddings

### Parsing Large Documents

Only HTML files of textbooks are accounted for at this time. Use the ```book_parser.py``` to convert your html file to csv or jsonl formats. If you want to convert to jsonl, add ``--jsonl``. 

```python scripts/book_parser.py path/to/html/file path/to/output```

### TF-IDF Embeddings 

We use TF-IDF embeddings to compare queries to sections. This helps us find relevant sections. To build our TF-IDF vectorizer and sparse matrix embeddings, use:

```python scripts/create_tfidf.py path/to/csv_or_db_file path/to/output```

## Training ALBERT

In this module, we train Albert on the [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset.



1. Parse the json files to create csv files 

These will be easier for our dataloaders to read. Use the following script:

```python squad_parser.py path/to/json/dir path/to/output/dir```

```path/to/json/dir``` should contain the train and dev json files from SQuAD. 

***FURTHER STEPS TO BE ADDED***
