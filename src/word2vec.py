from __future__ import print_function, division

import os
import numpy as np
from gensim.models import KeyedVectors

d = "../data"
w2v = KeyedVectors.load_word2vec_format(os.path.join(d, "wikipedia-pubmed-and-PMC-w2v.bin"), binary=True)
print("word2vec parameters loaded.")


def avg_sent_emb(s):
    words = s.split()
    l = len(words)
    
    for w in words:
        if w not in w2v:
            l -= 1
    
    return np.sum([w2v[w] for w in s.split() if w in w2v], axis=0) / l


def convert_sents_to_emb(sents):
    emb = []
    for p in sents:
        emb.append(np.array([avg_sent_emb(p[0]), avg_sent_emb(p[1])]))
    
    return np.array(emb)
