from __future__ import print_function, division

import util, word2vec

sents, anns = util.load_biosses()
print("Data loaded.")

emb = word2vec.convert_sents_to_emb(sents)

cos_sims = util.compute_all_cos_sim(emb)

print(util.compute_pearson_coeff(cos_sims, anns))

