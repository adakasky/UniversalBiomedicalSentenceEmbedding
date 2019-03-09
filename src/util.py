from __future__ import print_function, division

import os
import spacy
import codecs
import pandas
import numpy as np

from multiprocessing import Pool
from gensim.parsing.preprocessing import remove_stopwords, strip_non_alphanum
from scipy.spatial.distance import cosine

nlp = spacy.load('en_core_web_sm')


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


def mp_sent_seg(sent):
    data = []
    if type(sent).__name__ == 'str':
        p = nlp(sent)
        
        for s in p.sents:
            data.append(s.text.lower() + "\n")
    
    return data


def convert_pubmed_to_ospl(file='../data/query_result.csv', out='../data/query_result_ospl.txt', num_process=8):
    p = Pool(num_process)
    
    with codecs.open(file, 'rb', 'utf-8') as r, codecs.open(out, 'wb', 'utf-8') as w:
        reader = pandas.read_csv(r, escapechar='\\', chunksize=num_process)
        
        for line in reader:
            print(line['pmid'])
            sents = line.values[:, 1:3].flatten()
            
            for data in p.imap(mp_sent_seg, sents):
                w.write("".join(data))


if __name__ == '__main__':
    convert_pubmed_to_ospl()
