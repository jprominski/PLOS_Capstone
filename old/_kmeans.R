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
names(introdf) <- 'abstract'

write.csv(introdf, 'introdf.csv', row.names = FALSE)

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
intro_clean.tfidf = DocumentTermMatrix(intro_clean, control = list(weighting = weightTfIdf))
# (weighting = weightTfIdf)
# weightTfIdf(dtm, normalize = TRUE)

tf.99 = removeSparseTerms(intro_clean.tfidf, 0.99)
tf.99

rowTotals <- apply(tf.99 , 1, sum) #Find the sum of words in each Document
tfidf_sparse   <- tf.99[rowTotals>0,]


normalise_dtm<- function(y) y/apply(y, MARGIN=1, FUN=function(k) sum(k^2)^.5)
dtm_norm<- normalise_dtm(tfidf_sparse)

cl<- kmeans(dtm_norm, 8)
# Check the number of objects in each cluster
table(cl$cluster)
# Check the cluster assigned to each object 
cl$cluster
# check the center of each clusters
cl$centers
# Within cluster sum of squares by cluster
cl$withinss

### Cluster visualisation using 2 principal components
plot(prcomp(dtm_norm)$x, col=cl$cl)

cl4<- kmeans(dtm_norm, 4)
plot(prcomp(dtm_norm)$x, col=cl4$cl)

cl3<- kmeans(dtm_norm, 3)
cl3_pca <- prcomp(dtm_norm)
  
plot(cl3_pca$x, col=cl3$cl)


dtm <- DocumentTermMatrix(intro_clean)
dtm_0 <- dtm[rowTotals>0,]

findFreqTerms(dtm_0[cl3$cluster==1], 50)
inspect(reuters[which(cl$cluster==1)])


?findFreqTerms

# start_time <- Sys.time()
# end_time <- Sys.time()
# end_time - start_time

# Eigenvalues are the variances of the principal components.
# ?screeplot
# screeplot(cl3_pca, npcs=2)
?princomp
?prcomp
?eigen

# word cloud


