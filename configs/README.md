# Configuration Set Up

Configurations are json files which contain the following keys:

data_path = this is where your squad csv files `` and `` are stored e.g. "../data/SQuAD/2.0"
output_dir = e.g. "./models" # for storing model weights between epochs
task = e.g. "SQuAD"
squad_version = e.g. "2.0"
testing= e.g.True
data_reduction = e.g. 1000 # reduce df sizes by this amount while testing
seed = e.g. 2020
model = e.g. "albert-base-v2"
max_lr= e.g.3e-5
max_lr_last = e.g. 1e-4
phases = e.g. .3
optimizer= e.g."lamb" # choose between 'adam' or 'lamb'
epochs= e.g.1
use_fp16= e.g.False
recreate_ds= e.g.False
bs= e.g.4
effective_bs= e.g.4 # set this different from bs to determine gradient accumulation steps (i.e. effective_bs/bs)
max_seq_len= e.g.512
start_tok = e.g. "[CLS]"
end_tok = e.g. "[SEP]"
sep_tok = e.g. "[SEP]"
unk_tok_idx= e.g.1
sep_idx= e.g.3
pad_idx= e.g.0
feat_cols = e.g. ["question""paragraph"]
label_cols = e.g. ["idxs""is_impossible"]
adjustment = e.g. 2
save_checkpoint = e.g. True
load_checkpoint= e.g."2.0/base"
num_labels_clas = e.g. 2
clas_dropout_prob = e.g. .1
stats_update_freq = e.g. .1
