# albert-qa
Building a Question Answering system with Albert on SQuAD dataset

## Training

In this module, we will train Albert on the [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

0. Clone the reposiroty 

```git clone https://github.com/devkosal/albert-qa/ | cd albert-qa```

1. Parse the json files to create csv files 

These will be easier for our dataloaders to read. Use the following script:

```python squad_parser.py path/to/json/dir path/to/output/dir```

```path/to/json/dir``` should contain the train and dev json files from SQuAD. 

***FURTHER STEPS TO BE ADDED***
