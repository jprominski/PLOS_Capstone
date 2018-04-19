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
corpus = corpora.MmCorpus('/volume/models/corpus.mm')
dictionary = corpora.Dictionary.load('/volume/models/dictionary.dict')

print('training lsi...')
start_time = time.time()
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=10)
lsi.save('/volume/models/tax_model.lsi')
print("--- %s seconds ---" % (time.time() - start_time))

print('training lda...')
start_time = time.time()
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=10)
lda.save('/volume/models/tax_model.lda')
print("--- %s seconds ---" % (time.time() - start_time))