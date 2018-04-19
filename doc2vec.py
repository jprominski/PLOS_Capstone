import pandas as pd
import numpy as np
from time import time
import pickle

import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument,TaggedLineDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec

print('loading docs...')
start_time = time()
documents = [doc for doc in TaggedLineDocument('volume2/processed_body_docs.txt')]
print("--- %s seconds ---" % (time() - start_time))

#documents = []
#with open('/volume/processed_body_docs.txt') as f:
#    for line in f:
#        documents.append(TaggedLineDocument(line))

print('training doc2vec model...')
start_time = time()
model = Doc2Vec(documents, vector_size=200, window=5, min_count=5,workers=14, epochs=20)
print("--- %s seconds ---" % (time() - start_time))

print('saving model...')
np.save('volume2/new_models/body_features-w2v-200.npy',model.docvecs.doctag_syn0)
model.save('volume2/new_models/body_features-w2v-200.doc2vec')

print('complete!')