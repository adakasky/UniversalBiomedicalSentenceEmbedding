from __future__ import print_function, division
from src import util

import itertools
import numpy as np

import torch
from torch.autograd import Variable

from skipthoughts import UniSkip, DropUniSkip, BayesianUniSkip, BiSkip

dir_st = '../data/skip-thoughts'

sents, anns = util.load_biosses()
s1 = list(map(lambda p: p[0].split(), sents))
s2 = list(map(lambda p: p[1].split(), sents))
s1_len = [len(s) for s in s1]
s2_len = [len(s) for s in s2]
max_len = max(max(s1_len), max(s2_len))

v1 = set(list(itertools.chain(*s1)))
v2 = set(list(itertools.chain(*s2)))
vocab = list(v1.union(v2))
wtoi = dict([(w, i + 1) for i, w in enumerate(vocab)])

s1_toi = Variable(torch.LongTensor([([wtoi[w] for w in s] + [0] * max_len)[:max_len] for s in s1]))
s2_toi = Variable(torch.LongTensor([([wtoi[w] for w in s] + [0] * max_len)[:max_len] for s in s2]))

# model = UniSkip(dir_st, vocab)
# model = DropUniSkip(dir_st, vocab)
# model = BayesianUniSkip(dir_st, vocab)
model = BiSkip(dir_st, vocab)


s1_vec = model(s1_toi, lengths=s1_len).detach().numpy()[:, np.newaxis, :]
s2_vec = model(s2_toi, lengths=s2_len).detach().numpy()[:, np.newaxis, :]

emb = np.concatenate((s1_vec, s2_vec), axis=1)
cos_sims = util.compute_all_cos_sim(emb)

print(util.compute_pearson_coeff(cos_sims, anns))
