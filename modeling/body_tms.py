import sqlite3
import pandas as pd
import numpy as np
import time
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.moses import MosesDetokenizer
from stop_words import get_stop_words
from collections import defaultdict
import gensim
from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument,TaggedLineDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec


print('loading...')
bod_tok = pd.read_pickle('/volume/bodtokens.pkl')
texts = bod_tok.tolist()

print('generating dictionary and corpus...')
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print('saving...')
corpora.MmCorpus.serialize('/volume/models/bod_corpus.mm', corpus)
dictionary.save('/volume/models/bod_dictionary.dict')

print('training tfidf...')
start_time = time.time()
tfidf = models.TfidfModel(corpus, normalize=True)
print("--- %s seconds ---" % (time.time() - start_time))

print('saving...')
tfidf.save('/volume/models/bod_model.tfidf')

print('training lsi...')
start_time = time.time()
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=200)
lsi.save('/volume/models/bod_model.lsi')
print("--- %s seconds ---" % (time.time() - start_time))

print('training lda...')
start_time = time.time()
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=200)
lda.save('/volume/models/bod_model.lda')
print("--- %s seconds ---" % (time.time() - start_time))