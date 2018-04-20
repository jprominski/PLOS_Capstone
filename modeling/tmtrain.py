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

t = time.asctime(time.localtime(time.time()-14400))
print('-----------')
print('Time is: ' + t)


print('loading data...')
start_time = time.time()
conn = sqlite3.connect("volume2/testDB.db")
tdf = pd.read_sql('SELECT bod_toks FROM PLOS_all_tok', conn)
print("--- %s seconds ---" % (time.time() - start_time))


print('converting to list...')
start_time = time.time()
bodtexts = tdf['bod_toks'].tolist()
print("--- %s seconds ---" % (time.time() - start_time))


print('cleaning...')
start_time = time.time()
ls = []
for line in bodtexts:
    l = line.replace("'","")
    l = l.strip('[]')
    ls.append(l)
print("--- %s seconds ---" % (time.time() - start_time))
    

print('tokenizing...')
start_time = time.time()
texts = []
for i in range(0,len(ls)):
    texts.append(ls[i].split(', '))
print("--- %s seconds ---" % (time.time() - start_time)) 


print('Creating dictionary and corpus...')
start_time = time.time()
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print("--- %s seconds ---" % (time.time() - start_time))


print('Saving dictionary and corpus...')
start_time = time.time()
corpora.MmCorpus.serialize('volume2/new_models/body_corpus.mm', corpus)
dictionary.save('volume2/new_models/body_dictionary.dict')
print("--- %s seconds ---" % (time.time() - start_time))


print('Creating and saving tfidf...')
start_time = time.time()
tfidf = models.TfidfModel(corpus, normalize=True)
tfidf.save('volume2/new_models/body_model.tfidf')
print("--- %s seconds ---" % (time.time() - start_time))


print('training lsi model...')
start_time = time.time()
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=200)
lsi.save('volume2/new_models/body_model.lsi')
print("--- %s seconds ---" % (time.time() - start_time))


print('training lda model...')
start_time = time.time()
lda = models.LdaModel(corpus, id2word=dictionary, num_topics=200)
lda.save('volume2/new_models/body_model.lda')
print("--- %s seconds ---" % (time.time() - start_time))


print('creating index...')
start_time = time.time()
index = similarities.MatrixSimilarity(lsi[corpus])
print("--- %s seconds ---" % (time.time() - start_time))

print('saving index...')
index.save('volume2/new_models/body_similarity.index')

print('Complete!')
t = time.asctime(time.localtime(time.time()-14400))
print('Time is: ' + t)