import sqlite3
import pandas as pd
import numpy as np
import time
from tqdm import tqdm_notebook as tqdm
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from multiprocessing import Pool

def toker(doc):
    raw = doc.strip('Introduction')
    # clean and tokenize document string
    raw2 = raw.lower()
    toks = tokenizer.tokenize(raw2)
    # remove stop words from tokens
    stopped_tokens = [i for i in toks if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # strip numbers from tokens
    stripped_tokens = [i for i in stemmed_tokens if not i.isdigit()] 
    return stripped_tokens

def p_toker(data):
    data['bod_toks'] = data['body'].apply(lambda x: toker(x))
    return data

conn = sqlite3.connect("/volume/testDB.db")
start_time = time.time()
print('reading in df...')
df = pd.read_sql('SELECT doi,abstract,body from PLOS_ALL', conn)
print("--- %s seconds ---" % (time.time() - start_time))


tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()


print('tokenizing bodies...')


start_time = time.time()

cores = 14
partitions = cores #Define as many partitions as you want
 
def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

df = parallelize(df, p_toker)

# df['bod_toks'] = df['body'].apply(lambda x: toker(x))

print("--- %s seconds ---" % (time.time() - start_time))

print('converting tokens to string...')
df['bod_toks'] = df['bod_toks'].apply(lambda x: str(x))


# print('pickling...')
# start_time = time.time()
# df['bod_toks'].to_pickle('/volume/bodtokens.pkl')
# print("--- %s seconds ---" % (time.time() - start_time))

print('persisting to db...')
start_time = time.time()
if 'level_0' in df:
    df = df.drop(['level_0'],axis=1)
df.to_sql("PLOS_ALL_tok", conn, if_exists="replace")
print("--- %s seconds ---" % (time.time() - start_time))

print('Complete!')










