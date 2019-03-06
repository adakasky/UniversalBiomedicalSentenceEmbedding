from __future__ import print_function, division

import os
import spacy
import codecs
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


def convert_pubmed_to_ospl(file='../data/query_result.csv', out='../data/query_result_ospl.txt'):
    with codecs.open(file, 'rb', 'utf-8') as r, codecs.open(out, 'wb', 'utf-8') as w:
        nlp = spacy.load('en_core_web_sm')
        reader = pandas.read_csv(r, escapechar='\\', chunksize=1)
        
        for line in reader:
            title = line['title'].item()
            abstract = line['abstract'].item()
            
            if type(title).__name__ == 'str':
                w.write(title.lower() + "\n")
                
            if type(abstract).__name__ == 'str':
                p = nlp(abstract)
                
                for s in p.sents:
                    w.write(s.text.lower() + "\n")
            

convert_pubmed_to_ospl()
