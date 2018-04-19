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

print('loading data...')
conn = sqlite3.connect("/volume/testDB.db")
tdf = pd.read_sql('SELECT * FROM PLOS_all_tok', conn)

print('converting to list...')
bodtexts = tdf['bod_toks'].tolist()

print('splitting strings...')
#split strings
ls = []

for i in range(0,len(bodtexts)):
    ls.append(bodtexts[i].split(', '))
    
    
#for doc in bodtexts:
#    ls.append(doc.split(', '))



print('retokenizing...')
#re-tokenize
texts = []
for line in ls:
    l = line.replace("'","")
    l = l.strip('[]')
    texts.append(l)

print('Creating dictionary and corpus...')
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print('Saving dictionary and corpus...')
corpora.MmCorpus.serialize('/volume/models/body_corpus.mm', corpus)
dictionary.save('/volume/models/body_dictionary.dict')

tfidf = models.TfidfModel(corpus, normalize=True)
tfidf.save('/volume/body_model.tfidf')

print('training lsi model...')
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=200)

print('saving lsi model...')
lsi.save('/volume/models/body_model.lsi')

print('creating index...')
index = similarities.MatrixSimilarity(lsi[corpus])

print('saving index...')
index.save('/volume/models/body_similarity.index')