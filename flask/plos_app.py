import sqlite3
import pandas as pd
from flask import Flask, render_template, request
from gensim import corpora, models, similarities
from gensim.models import Doc2Vec
import time
import os
from random import randint 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", title="PLOS Capstone")

@app.route('/', methods=['GET', 'POST'])
def rand():
    
    data = {}
    x = randint(0,len(df.index))
    data['random'] = df.loc[x]
    
    #df.loc[x][['doi','abstract']]
    
    return render_template('random.html', **data)

@app.route('/dresults', methods=['GET', 'POST'])
def runDOI():

    data = {}

    ## search based on DOI -- return similar documents for LSI and D2V
    query_doi = request.form['doi_in']
    query_doc = df.loc[df['doi']==query_doi].index[0]

    data['target'] = df.loc[df['doi']==query_doi]
    
    # Generate similar documents - LSI
    doc_bow = corpus[query_doc]
    vec_lsi = lsi[doc_bow]
    lsi_sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
    simdocs_lsi = [x[0] for x in lsi_sims[1:11]]

    data['lsioutput'] = df.loc[simdocs_lsi]

    # Generate similar documents - D2V
    d2vsims = d2vmodel.docvecs.most_similar(query_doc, topn=10)
    d2vsimdocs = [x[0] for x in d2vsims]   
    
    data['d2voutput'] = df.loc[d2vsimdocs]

    return render_template('sim_doi_results.html', **data)



@app.route('/qresults', methods=['GET', 'POST'])
def runQuery():
    
    data = {}
    query_abs = request.form['query_in']
    
    vec_bow = dictionary.doc2bow(query_abs.lower().split())
    vec_lsi = lsi[vec_bow]
    lsi_sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
    simeds_lsi = [x[0] for x in lsi_sims[0:10]]

    data['lsioutput'] = df.loc[simeds_lsi]
    data['query'] = query_abs
    
    q_tokens = "query_in".split()
    q_vector = d2vmodel.infer_vector(q_tokens)    
    d2vsims = d2vmodel.docvecs.most_similar([q_vector], topn=10)
    d2vsimdocs = [x[0] for x in d2vsims]   
    
    data['d2voutput'] = df.loc[d2vsimdocs]

    return render_template('sim_q_results.html', **data)



@app.route('/edresults', methods=['GET', 'POST'])
def edRecs():

    ##### Results based on single most similar article
    
    data = {}
    query_abs = request.form['ab_in']
    data['query'] = query_abs

    vec_bow = dictionary.doc2bow(query_abs.lower().split())
    vec_lsi = lsi[vec_bow]
    lsi_sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
    simeds_lsi = [x[0] for x in lsi_sims[0:10]]

    ed_recs = df.loc[simeds_lsi]
    ed_recs['editors'] = ed_recs['editors'].str.strip("[]'")
    
    data['lsioutput'] = ed_recs
    
    q_tokens = query_abs.split()
    q_vector = d2vmodel.infer_vector(q_tokens)    
    d2vsims = d2vmodel.docvecs.most_similar([q_vector], topn=10)
    d2vsimdocs = [x[0] for x in d2vsims]   
    
    d2vrecs = df.loc[d2vsimdocs]
    d2vrecs['editors'] = d2vrecs['editors'].str.strip("[]'")
    
    data['d2voutput'] = d2vrecs
    
    ##### Results based on editor vector
        
    ed_d2vsims = ed_d2vmodel.docvecs.most_similar([q_vector], topn=10)
    ed_d2vsimdocs = [x[0] for x in ed_d2vsims]   
    
    ed_d2vrecs = df.loc[ed_d2vsimdocs]
    ed_d2vrecs['editors'] = ed_d2vrecs['editors'].str.strip("[]'")
    
    data['ed_d2voutput'] = ed_d2vrecs
    
    return render_template('ed_results.html', **data)


# Load models
start_time = time.time()
print('loading in models...')
lsi = models.LsiModel.load('/volume/models/model.lsi')
index = similarities.MatrixSimilarity.load('/volume/models/similarity.index')
corpus = corpora.MmCorpus('/volume/models/corpus.mm')
dictionary = corpora.Dictionary.load('/volume/models/dictionary.dict')
tfidf = models.TfidfModel.load('/volume/models/model.tfidf')
d2vmodel = Doc2Vec.load('/volume/models/features-w2v-200.doc2vec')


ed_lsi = models.LsiModel.load('/volume/new_models/editor_model.lsi')
ed_index = similarities.MatrixSimilarity.load('/volume/new_models/editor_similarity.index')
ed_corpus = corpora.MmCorpus('/volume/new_models/editor_corpus.mm')
ed_dictionary = corpora.Dictionary.load('/volume/new_models/editor_dictionary.dict')
ed_tfidf = models.TfidfModel.load('/volume/new_models/editor_model.tfidf')
ed_d2vmodel = Doc2Vec.load('/volume/new_models/editor_features-w2v-200.doc2vec')




# Transform corpus
corpus_tfidf = tfidf[corpus]
corpus_lsi = lsi[corpus_tfidf]

# Read in db
print('reading in data...')
conn = sqlite3.connect("/volume/testDB.db")
df = pd.read_sql('SELECT doi,title,editors,abstract from PLOS_ALL', conn)
print("--- %s seconds ---" % (int(time.time() - start_time)))
conn.close()


if __name__ == '__main__':
    app.debug = True 
    app.run(use_reloader=False,
           host=os.getenv('LISTEN', '0.0.0.0'))