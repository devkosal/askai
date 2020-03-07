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
from src import AlbertForQuestionAnsweringMTL
from src.utils_app import get_pred

app = Flask(__name__)
api = Api(app)

model = AlbertForQuestionAnsweringMTL.from_pretrained("models/2.0/base/2") # ensure pytroch_model.bin and config files are saved in directory
model.eval()
tok = AutoTokenizer.from_pretrained("albert-base-v2")

reqparser = reqparse.RequestParser()
reqparser.add_argument('texts',action="append")
reqparser.add_argument('question')

class Model(Resource):
    def get(self):
        reqargs = reqparser.parse_args()
        texts = reqargs["texts"]
        question = reqargs["question"]

        assert type(texts) == list, "input texts are not of type list"
        assert type(question) == str, "input question is not a string"
        pred, best_section = get_pred(texts, question, model, tok, pad_idx=0)

        return {"pred":pred,"best_section":best_section}

##
## Actually setup the Api resource routing here
##
api.add_resource(Model, '/')

if __name__ == '__main__':
    app.run(debug=True)
