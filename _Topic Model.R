library(XML)
library(tidyverse)
library(tm)
library(topicmodels)

#10,000 articles from PLOS One

# Get filenames of XML files in corpus
setwd("~/Desktop/UVA DSI/Capstone/")
filenames <- dir('sample_10000')[2:10001]
setwd("~/Desktop/UVA DSI/Capstone/sample_10000")

# Extract abstracts from XML files and convert to df
intro <- 0
for(i in 1:length(filenames)) {
  doc <- xmlTreeParse(filenames[i], useInternal=TRUE)
  top <- xmlRoot(doc)
  intro[i] <- xmlValue(top[[3]][['sec']])
}
introdf <- as.data.frame(intro, stringsAsFactors = FALSE)

# Clean data - remove Introduction from beginning of document
introdf <- as.data.frame(sub('Introduction*', "", introdf$intro))

# Convert to corpus
intro_corp = VCorpus(DataframeSource(introdf))

# Clean data
intro_clean = tm_map(intro_corp, stripWhitespace)     
intro_clean = tm_map(intro_clean, removeNumbers)                
intro_clean = tm_map(intro_clean, removePunctuation)             
intro_clean = tm_map(intro_clean, content_transformer(tolower)) 
intro_clean = tm_map(intro_clean, removeWords, stopwords("english"))
intro_clean = tm_map(intro_clean, stemDocument)  

# Compute TF Matrix
intro_clean.tf = DocumentTermMatrix(intro_clean, control = list(weighting = weightTf))
# (weighting = weightTfIdf)

tf.99 = removeSparseTerms(intro_clean.tf, 0.99)
tf.99

rowTotals <- apply(tf.99 , 1, sum) #Find the sum of words in each Document
tf_sparse   <- tf.99[rowSums(tf.99)>0,]




# Train topic models
topic.model2 = LDA(tf_sparse, 2)
topic.model5 = LDA(tf_sparse, 5)
topic.model10 = LDA(tf_sparse, 10)
topic.model25 = LDA(tf_sparse, 25)
topic.model50 = LDA(tf_sparse, 50)
topic.model100 = LDA(tf_sparse, 100)

# look at the top 10 words within the first 5 topics
terms(topic.model2, 10)
terms(topic.model5, 10)
terms(topic.model10, 10)[,1:10]
terms(topic.model25, 10)[,1:10]
#terms(topic.model50, 10)[,1:20]
#terms(topic.model100, 10)[,1:5]

# Thematically Coherent Topics
terms(topic.model10, 10)[,c(2,3,5,8)]


