#use IncidentDescription to predict risk rating level of an incident

#install.packages("tm")
#install.packages("SnowballC") #tm uses SnowballC for stemming
install.packages("quanteda")
install.packages("quanteda.textplots")
install.packages("tidyverse")
install.packages("kernlab")
install.packages("e1071")
install.packages("stm")
#install.packages("text2vec")

library(stm)
library(tidyverse)
library(quanteda) #https://quanteda.io/articles/quickstart.html
library(quanteda.textplots)
library(kernlab)      # SVM 
library(e1071) 
#library(text2vec)


setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
incidents <- read.csv("incidents.translated.csv")
data <- incidents %>% select(IncidentDescription, RiskRating)

#deduplicate: 37 rows are duplicated
data <-data[!duplicated(data),]

sum(is.na(data$RiskRating))
#create document-feature dataframe with labels
corp <- corpus(data, text_field="IncidentDescription") 
docvars(corp)
head(tokeninfo <- summary(corp)) #types=text word Tokens + punct
tokens <-tokens(corp, remove_punct = TRUE, remove_number=TRUE)

#key word in context

kwic(tokens, pattern="tom", valuetype = "regex")
kwic(tokens, pattern=phrase("full of water"))

dfm <- dfm(tokens_tolower(tokens)) # 4,963 documents, 9,717 features
dfm <- dfm_remove(dfm, c(stopwords("english"), "employee", "ee", "mine"))
dfm <- dfm_remove(dfm, pattern="^[a-z]$", valuetype="regex")
dfm <- dfm_remove(dfm, pattern="[0-9]", valuetype="regex")
dfm # 4,963 documents, 6921 features
#trim and tf*idf scoring 
dfm <- dfm_trim(dfm, min_termfreq = 5,  termfreq_type="count", 
                     max_docfreq = 0.50, docfreq_type="prop")
dfm #4,963 documents, 2403 features
dfm@Dimnames[["features"]]
c("water", "we", "a", "on", "to", "at", "employee" )%in% dfm@Dimnames[["features"]]
#dfm[, 1:5]
#paste(dfm[1, ])
#paste(c("features of ", dfm[1, ]@Dimnames[["docs"]], " [", dfm[1, ], "]"), collapse =" ")
      
#all texts with all zeros in their features, should not be many
count <- 0
for(i in 1:nrow(dfm)){
  if(sum(dfm[i,]) == 0){
    print(dfm[i,]@Dimnames[["docs"]])
    count <- count+1
  }
}
paste(c(count, "files without any features"), collapse = " ")
set.seed(100)
textplot_wordcloud(dfm, min_count = 1, random_order = FALSE, rotation = 0.25, 
                         color = RColorBrewer::brewer.pal(8, "Dark2"))
      
"water" %in% dfm@Dimnames[["features"]]
dfm@Dimnames[["docs"]]

set.seed(100)
if (require("stm")) { #work with counts
  my_lda_fit20 <- stm(dfm, K = 20, verbose = FALSE)
  plot(my_lda_fit20)
}


dfm<-dfm_tfidf(dfm) %>% round(digits = 2)
dfm





#matrix<-convert(dfm(tokens), to="data.frame"))
df <-convert(dfm, to="data.frame")
#explictly set levels to avoid inconsistency btw train and test in tune
df$RiskRating <- factor(data$RiskRating, levels=c("Critical","High","Low","Moderate"))
df<-df[,-1]
#run svm
grep("RiskRating", colnames(df))
#use simple term frequency
model <- svm(RiskRating~., data = df, kernel = "polynomial", gamma=0.1, type="C-classification", na.action = na.omit, cross=5)
model$accuracies # 59.87903 61.53072 61.18952 62.03424 58.71098
#linear: 9717: 59.77823 58.40886 61.89516 58.91239 60.12085
#radial: 9717: 66.43145 64.24975 65.42339 68.37865 65.35750
#        2662 features:  65.22177 65.65962 67.43952 63.94763 67.47231


#takes 20 hrs
tune.out <- tune(svm, RiskRating~., data = df, kernel = "radial", type="C-classification", na.action=na.omit,
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5), 
                               gamma = c(0.001, 0.01, 0.1, 1, 5)
                 ))
tune.out$best.parameters # gamma 0.001, cost 5
tune.out$best.performance #error rate, the lower the better 0.33 

tune.out <- tune(svm, RiskRating~., data = df, kernel = "radial", type="C-classification", na.action=na.omit,
                 ranges = list(cost = c(3, 5, 7), 
                               gamma = c(0.0001, 0.005, 0.001)
                 ))

tune.out$best.parameters # gamma 0.001, cost 5
tune.out$best.performance #error rate, the lower the better 0.33 

tune.out <- tune(svm, RiskRating~., data = df, kernel = "polynomial", type="C-classification", na.action=na.omit,
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5), 
                               gamma = c(0.001, 0.01, 0.1, 1, 5), 
                               degree= c(2,3,4)
                 ))
tune.out$best.parameters #28  0.1 0.001      3
tune.out$best.performance #error rate, the lower the better 0.3386995

#use word embedding
#how to train an embeding using word2vec
#https://cran.r-project.org/web/packages/text2vec/vignettes/glove.html
#https://rpubs.com/nabiilahardini/word2vec
#https://medium.com/broadhorizon-cmotions/using-word-embedding-models-for-prediction-purposes-34b5bc93c6f
#R.utils package is used for unpacking pre-trained word embedding models. 
#Keras package. 





#convert description text to document by term matrix
#for(i in 1:nrow(data)){
#  data[i, "IncidentDescription"] <- SimpleCorpus(VectorSource(data[i, "IncidentDescription"])) %>%
#      tm_map(removePunctuation) %>%
#      tm_map(content_transformer(tolower)) %>%
#      tm_map(removeWords, stopwords("en")) %>%
#      tm_map(stripWhitespace) %>%
#      tm_map(stemDocument) #%>%
      #content()
#}


#corpus <- SimpleCorpus(VectorSource(data[, "IncidentDescription"]))
#corpus <- corpus %>%
#  tm_map(removePunctuation) %>%
#  tm_map(content_transformer(tolower)) %>%
#  tm_map(removeWords, stopwords("en")) %>%
#  tm_map(stripWhitespace) %>%
#  tm_map(stemDocument)

#DT <-DocumentTermMatrix(corpus, control=list(weighting=weightTfIdf))

#dfm(tokens(corpus(corpus)))


