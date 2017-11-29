# Topic Modeling HW
# Jack Prominski
# jmp3mk

library(XML)
library(tidyverse)
library(tm)
library(topicmodels)

# The data source is 189 articles from the scientific journal PLOS ONE. 
# Hypothesized topic labels: Cancer, Clinical trials, Genetics, Cell biology


# Get filenames of XML files in corpus
setwd("~/Desktop/UVA DSI/SYS6018")
filenames <- dir('Corpus')[3:191]
setwd("~/Desktop/UVA DSI/SYS6018/Corpus")

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

# Train topic models
topic.model2 = LDA(intro_clean.tf, 2)
topic.model5 = LDA(intro_clean.tf, 5)
topic.model10 = LDA(intro_clean.tf, 10)
topic.model25 = LDA(intro_clean.tf, 25)
topic.model50 = LDA(intro_clean.tf, 50)
topic.model100 = LDA(intro_clean.tf, 100)

# look at the top 10 words within the first 5 topics
terms(topic.model2, 10)
terms(topic.model5, 10)
terms(topic.model10, 10)[,1:10]
terms(topic.model25, 10)
terms(topic.model50, 10)[,1:20]
terms(topic.model100, 10)[,1:5]

# Thematically Coherent Topics
terms(topic.model50, 10)[,c(7,17)]
terms(topic.model25, 10)[,c(2,6)]

# Thematically Coherent Topics
# Topic 2    Topic 6   
# [1,] "plant"    "ahr"     
# [2,] "pathogen" "cell"    
# [3,] "interact" "gene"    
# [4,] "soil"     "spine"   
# [5,] "mucos"    "protein" 
# [6,] "studi"    "dendrit" 
# [7,] "process"  "actin"   
# [8,] "can"      "interact"
# [9,] "model"    "tcr"     
# [10,] "diseas"   "antigen" 

# Topic 7   Topic 17  
# [1,] "use"     "gene"    
# [2,] "bone"    "drug"    
# [3,] "fractur" "resist"  
# [4,] "model"   "method"  
# [5,] "patient" "associ"  
# [6,] "facial"  "mutat"   
# [7,] "implant" "cancer"  
# [8,] "mbf"     "nsclc"   
# [9,] "minipl"  "cluster" 
# [10,] "plate"   "identifi"

