# based on docs @ https://flask-restful.readthedocs.io/en/latest/quickstart.html

import os
try:
    get_ipython
    os.chdir("/Users/devsharma/Dropbox/Projects/tbqa/askai")
except:
    pass

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from transformers import AutoTokenizer, PretrainedConfig
from src.model import AlbertForQuestionAnsweringMTL
from src.utils_backend import get_pred

import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import load_npz
import pickle
from pathlib import Path

from src.utils_app import get_contexts, get_scores, bold_answer, Config

import pandas as pd
import re
import json
import sys
from requests import get


app = Flask(__name__)
api = Api(app)

example = "health_education"  # default

# setting and loading configuration variables
config = Config(
    model="albert-base-v2",
    pad_idx=0,
    weights="models/2.0/base/2",
    **json.load(open(f'examples/{example}/book-config.json', "r"))
)

# ensure pytroch_model.bin and config files are saved in directory
model = AlbertForQuestionAnsweringMTL.from_pretrained(config.weights)
model.eval()
tok = AutoTokenizer.from_pretrained(config.model)

reqparser = reqparse.RequestParser()
reqparser.add_argument('question')

# determine the data type (whether csv or db)
if config.sections_file_type == "db":
    # connecting to the DB
    con = sqlite3.connect(
        f'examples/{example}/sections.{config.sections_file_type}', check_same_thread=False)
    data = con.cursor()
elif config.sections_file_type == "csv":
    data = pd.read_csv(
        f'examples/{example}/sections.{config.sections_file_type}')

# load vectors and vectorizer
X = load_npz(f"examples/{example}/tfidf-vectors.npz")
vectorizer = pickle.load(open(f"examples/{example}/vectorizer.pkl", "rb"))


class Model(Resource):
    def post(self):
        reqargs = reqparser.parse_args()
        question = reqargs["question"]
        assert type(question) == str, "input question is not a string"

        # get scored sections in descending order
        scores = get_scores(question, vectorizer, X)
        # get the most relevant sections' raw texts
        contexts = get_contexts(scores, data)
        # get answer, most relevant text
        pred, best_section = get_pred(
            contexts, question, model, tok, pad_idx=config.pad_idx)

        return {"pred": pred, "best_section": best_section}


class Test(Resource):
    def get(self):
        return {
            "tests":
            [
                {"nm": "test item 1", "i": 0},
                {"nm": "test item 2", "i": 1}
            ]
        }


##
# Actually setup the Api resource routing here
##
api.add_resource(Model, '/api')
api.add_resource(Test, "/tests")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
