# Configuration Set Up

Configurations are json files which contain the following keys:
- **data_path**: where your squad csv files are stored from running `squad_parser.py` e.g. "../data/SQuAD/2.0"
- **output_dir**: where your models are stored after epochs e.g. "./models"
- **task**: task name e.g. "SQuAD"
- **squad_version**: the version of squad data e.g. "2.0"
- **testing**: whether you want to test on a smaller subset of data (weights are not stored when testing) e.g. True
- **data_reduction**: reduce csv sizes by this amount while testing e.g. 1000 
- **seed**: random seed e.g. 2020
- **model**: the model name from huggingface's transformers e.g. "albert-base-v2"
- **max_lr**: max learning rate for base albert model e.g. 3e-5
- **max_lr_last**: max learning rate for final layers e.g. 1e-4
- **phases**: the peak for learning rate annealing e.g. .3
- **optimizer**: choose between 'adam' or 'lamb' e.g. "lamb" 
- **epochs**: e.g. 1
- **use_fp16**: e.g. False
- **recreate_ds**: datasets are pickled for faster retraining. setting this to true will recreate the dataset e.g. False
- **bs**: batchsize e.g. 4
- **effective_bs**: set this different from bs to determine gradient accumulation steps (i.e. effective_bs/bs) e.g. 4 
- **max_seq_len**: max sequence length e.g. 512
- **start_tok**: start of sequence token e.g. "[CLS]"
- **end_tok**: end of sequence token e.g. "[SEP]"
- **sep_tok**: seperation token e.g. "[SEP]"
- **unk_idx**: the idx of the unkown token (1 for albert) e.g. 1
- **sep_idx**: the idx of the seperation token (1 for albert) e.g.3
- **pad_idx**: the idx of the pad token (1 for albert) for pad collating e.g. 0
- **feat_cols**: feature columns e.g. ["question""paragraph"]
- **label_cols**: label columns e.g. ["idxs""is_impossible"]
- **adjustment**: the amount to adjust answer spans by, typically the number of special tokens added to model input e.g. 2
- **save_checkpoint**: whether checkpoints are saved e.g. True
- **load_checkpoint**: loads existing checkpoint e.g. "2.0/base"
- **num_labels_clas**: number of labels for 'is_impossible' label e.g. 2
- **clas_dropout_prob**: the dropout probability for final layer of 'is_impossible' classifier e.g. .1 
- **stats_update_freq**: how frequently stat updates occur e.g. .1

