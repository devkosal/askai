
        #################################################
        ### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
        #################################################
        # file to edit: notebooks/Train Albert on SQuAD.ipynb

import os

from transformers import AutoTokenizer,PretrainedConfig
import numpy as np
import pandas as pd
import pickle
import re
import requests
import json
from src import *
import fire

def load_dfs(config):
    train = pd.read_csv(config.data_path+f"/train_{config.squad_version}_{config.model_name}.csv")
    valid = pd.read_csv(config.data_path+f"/val_{config.squad_version}_{config.model_name}.csv")

    train.drop_duplicates(inplace=True)
    valid.drop_duplicates(inplace=True)

    # randomizing the order of training data
    train = train.sample(frac=1).reset_index(drop=True) #random_state = config.seed
    valid = valid.sample(frac=1).reset_index(drop=True)

    # reduce df sizes if testing
    if config.testing:
        train = train[:int(len(train)/config.data_reduction)]
        valid = valid[:int(len(valid)/config.data_reduction)]

    return remove_max_sl(train, config.max_seq_len), remove_max_sl(valid, config.max_seq_len)

def make_dataloaders(config, train_df, valid_df):
    tok = AutoTokenizer.from_pretrained(config.model)

    proc_tok = QATokenizerProcessor(tok.tokenize, config.max_seq_len, config.start_tok, config.end_tok)

    vocab = {tok.convert_ids_to_tokens(i):i for i in range(tok.vocab_size)}
    proc_num = QANumericalizeProcessor(vocab, unk_tok_idx=config.unk_idx)
    proc_qa = QALabelProcessor(str2tensor,config.adjustment)

    if (not (os.path.exists(config.data_path+f"/squad_{config.squad_version}_data_trn.pkl"))) or config.recreate_ds or config.testing:
        il_train = SquadTextList.from_df(train_df,config.feat_cols,config.label_cols,config.sep_tok)
        il_valid = SquadTextList.from_df(valid_df,config.feat_cols,config.label_cols,config.sep_tok)

        ll_valid = LabeledData(il_valid,il_valid.labels,proc_x = [proc_tok,proc_num], proc_y=[proc_qa])
        ll_train = LabeledData(il_train,il_train.labels,proc_x = [proc_tok,proc_num], proc_y=[proc_qa])

        # saving/loading presaved data if not testing
        if not config.testing:
            # save an object
            pickle.dump(ll_train, open( config.data_path+f"/squad_{config.squad_version}_data_trn.pkl", "wb" ) )
            pickle.dump(ll_valid, open( config.data_path+f"/squad_{config.squad_version}_data_val.pkl", "wb" ) )
    else:
        # load an object
        ll_train = pickle.load( open( config.data_path+f"/squad_{config.squad_version}_data_trn.pkl", "rb" ) )
        ll_valid = pickle.load( open( config.data_path+f"/squad_{config.squad_version}_data_val.pkl", "rb" ) )

    collate_fn = partial(pad_collate_qa,pad_idx=config.pad_idx)

    train_sampler = SortishSampler(ll_train.x, key=lambda t: len(ll_train[int(t)][0]), bs=config.bs)
    train_dl = DataLoader(ll_train, batch_size=config.bs, sampler=train_sampler, collate_fn=collate_fn)

    valid_sampler = SortSampler(ll_valid.x, key=lambda t: len(ll_valid[int(t)][0]))
    valid_dl = DataLoader(ll_valid, batch_size=config.bs, sampler=valid_sampler, collate_fn=collate_fn)

    return DataBunch(train_dl,valid_dl)

def get_learner(config, data, opt_func):
    model_kwargs = {"pretrained_model_name_or_path": config.weights}

    if not config.load_checkpoint: model_kwargs["askai_config"] = config
    model = AlbertForQuestionAnsweringMTL.from_pretrained(**model_kwargs)

    # setting up callbacks
    cbfs = [partial(QAAvgStatsCallback,[acc_qa,acc_pos,exact_match,f1_score]),
            ProgressCallback,
            Recorder]

    if torch.cuda.is_available(): cbfs.append(CudaCallbackMTL)

    if not config.testing and config.save_checkpoint:
        cbfs.append(partial(SaveModelCallback,save_model_qa,config.output_dir,config.model,config.squad_version))

    if config.effective_bs and config.bs != config.effective_bs:
        cbfs.append(partial(GradientAccumulation,config.bs,config.effective_bs))

    if config.stats_update_freq is not None: cbfs.append(partial(TrainStatsCallback,config.stats_update_freq))

    learn = Learner(model, data, cross_entropy_qa_mtl,lr=config.max_lr,cb_funcs=cbfs,splitter=albert_splitter,\
                opt_func=opt_func)
    return learn

def main(config):
    if isinstance(config, str): config = Config(**json.load(open(config,"r")))
    assert type(config) == Config, f"config parameter type must be Config or a path to a json file"
    if config.effective_bs:
        assert config.effective_bs >= config.bs, f"mini bs ({config.bs}) cannot be smaller than effective bs ({config.effective_bs})"
        assert config.effective_bs % config.bs == 0, "mini bs ({config.bs}) should be a factor of the effective bs ({config.effective_bs})"

    config.model_name=re.findall(r"(.+?)-",config.model)[0]
    config.weights=config.output_dir+f"/{config.load_checkpoint}" if config.load_checkpoint else config.model

    train,valid = load_dfs(config)
    data = make_dataloaders(config, train, valid)

    # set LR scheduler
    disc_lr_sched = sched_1cycle([config.max_lr,config.max_lr_last], config.phases)

    # set optimizer
    assert config.optimizer.lower() in ["adam","lamb"], f"invalid optimizer in config {config.optimizer}"
    opt_func = lamb_opt() if config.optimizer.lower() == "lamb" else adam_opt()

    learn = get_learner(config, data, opt_func)
    learn.fit(config.epochs,cbs=disc_lr_sched)

if __name__ == "__main__":
    fire.Fire(main)