import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz
import numpy as np
import pickle
from scipy import stats
import os
from pathlib import Path
import logging
import fire

def create_vectors(path_to_csv, output_dir=None):
    if output_dir is None: output_dir = Path(path_to_csv).parent
    df = pd.read_csv(path_to_csv)
    lens = [len(i.split()) for i in df.text]
    logger.info(f"stats: {stats.describe(lens)}")
    logger.info(f"total number of sections over 500 words:{sum([i > 500 for i in lens])}")

    vectorizer = TfidfVectorizer(stop_words="english",ngram_range=(1, 2),min_df=1)
    X = vectorizer.fit_transform(df.text,)
    logger.info(f"saving vectors as tfidf-vectors.npz in {output_dir}")
    save_npz("tfidf-vectors.npz", X)
    logger.info(f"saving vectorizer as vectorizer.pkl in {output_dir}")
    pickle.dump(vectorizer,open("vectorizer.pkl","wb"))

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fire.Fire(create_vectors)
