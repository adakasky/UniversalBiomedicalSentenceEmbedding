from __future__ import print_function, division

import os
import pandas
import numpy as np

from gensim.parsing.preprocessing import remove_stopwords, strip_non_alphanum
from scipy.spatial.distance import cosine


def load_biosses(d='../data'):
    sent_file = os.path.join(d, "BIOSSES_sents.tsv")
    ann_file = os.path.join(d, "BIOSSES_ann.tsv")
    sents = np.array(pandas.read_csv(sent_file, sep='\t'))[:, 1:]
    anns = np.average(np.array(pandas.read_csv(ann_file, sep='\t'))[:, 1:], axis=1)
    
    for p in sents:
        p[0] = strip_non_alphanum(remove_stopwords(str.lower(p[0])))
        p[1] = strip_non_alphanum(remove_stopwords(str.lower(p[1])))
    
    return sents, anns / 4


def cos_sim(a, b):
    return 1 - cosine(a, b)


def compute_all_cos_sim(emb):
    result = np.zeros(emb.shape[0])
    
    for i, p in enumerate(emb):
        result[i] = cos_sim(p[0], p[1])
    
    return result


def compute_pearson_coeff(a, b):
    return np.corrcoef(a, b)[0, 1]
