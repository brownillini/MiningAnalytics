#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)

install.packages("fastText")
library(fastText)
library(stm)
library(dplyr)
library(arules) #mine rules with one item on the right hand side (rhs).
library(stringr)
library(tm)
library(stopwords)

#use MSHA inj_body_part data to train a model using fastText
#use the model to predict which body part injury an interaction/audit could be a leading factor

#https://towardsdatascience.com/fasttext-bag-of-tricks-for-efficient-text-classification-513ba9e302e7
#https://fasttext.cc/docs/en/supervised-tutorial.html
#using an example data set: data
setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1")




#guarding <- raw1[grepl("guarding", raw1$NARRATIVE), c(11, 12)]
#sort(table(guarding$INJ_BODY_PART))/sum(table(guarding$INJ_BODY_PART))


#merge labels 
data <- raw1 %>% select(NARRATIVE, INJ_BODY_PART) %>% 
  mutate(INJ_BODY_PART = case_when (
    INJ_BODY_PART == "EYE(S) OPTIC NERVE/VISON" ~ "EYE",
    INJ_BODY_PART == "HAND (NOT WRIST OR FINGERS)" ~ "HAND",
    INJ_BODY_PART == "FINGER(S)/THUMB" ~ "FINGER",
    INJ_BODY_PART == "WRIST" ~ "WRIST",
    INJ_BODY_PART == "ANKLE" ~  "ANKLE",
    INJ_BODY_PART == "KNEE/PATELLA" ~ "KNEE",
    INJ_BODY_PART == "SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA)" ~ "SHOULDER",
    INJ_BODY_PART == "BACK (MUSCLES/SPINE/S-CORD/TAILBONE)" ~ "BACK",
    INJ_BODY_PART == "FOREARM/ULNAR/RADIUS" ~ "FOREARM",
    INJ_BODY_PART ==  "ABDOMEN/INTERNAL ORGANS"~ "ABDOMEN",
    INJ_BODY_PART ==  "HIPS (PELVIS/ORGANS/KIDNEYS/BUTTOCKS)"~ "HIP",
    INJ_BODY_PART ==  "ELBOW" ~ "ELBOW",
    INJ_BODY_PART ==  "FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS"~ "FOOT",
    INJ_BODY_PART ==  "MOUTH/LIP/TEETH/TONGUE/THROAT/TASTE"~ "MOUTH",
    INJ_BODY_PART ==  "SCALP" ~ "SCALP",
    INJ_BODY_PART ==  "CHEST (RIBS/BREAST BONE/CHEST ORGNS)"~ "CHEST",
    INJ_BODY_PART ==  "LOWER LEG/TIBIA/FIBULA"~ "LLEG",
    INJ_BODY_PART ==  "NECK"~ "NECK",
    INJ_BODY_PART ==  "JAW INCLUDE CHIN" ~ "JAW",
    INJ_BODY_PART ==  "TOE(S)/PHALANGES" ~ "TOE",
    INJ_BODY_PART ==  "EAR(S) INTERNAL & HEARING" ~ "EAR",
    INJ_BODY_PART ==  "UPPER ARM/HUMERUS"~ "UPARM",
    INJ_BODY_PART ==  "BRAIN" ~ "BRAIN",
    INJ_BODY_PART ==  "THIGH/FEMUR"  ~ "THIGH",
    INJ_BODY_PART ==  "NOSE/NASAL PASSAGES/SINUS/SMELL"  ~ "NOSE",
    INJ_BODY_PART ==  "EAR(S) EXTERNAL"~ "EAR",
    INJ_BODY_PART ==  "SKULL"~ "SKULL",
    INJ_BODY_PART ==  "EAR(S) INTERNAL & EXTERNAL" ~ "EAR",

    INJ_BODY_PART ==  "BODY SYSTEMS"~ "BODY",
    INJ_BODY_PART ==  "MULTIPLE PARTS (MORE THAN ONE MAJOR)"~ "MULTIPLE",
    INJ_BODY_PART ==  "TRUNK, MULTIPLE PARTS" ~ "TRUNK",
    INJ_BODY_PART ==  "UPPER EXTREMITIES, MULTIPLE"~ "UPEXTREMITY",
    INJ_BODY_PART ==  "LOWER EXTREMITIES, MULTIPLE PARTS"~ "LWEXTREMITY",
    INJ_BODY_PART ==  "FACE, MULTIPLE PARTS" ~ "FACE",
    INJ_BODY_PART ==  "ARM, MULTIPLE PARTS" ~ "ARM",
    INJ_BODY_PART ==  "HEAD, MULTIPLE PARTS"~ "HEAD",
    INJ_BODY_PART ==  "LEG, MULTIPLE PARTS" ~ "LEG",
    
    INJ_BODY_PART ==  "FACE,NEC" ~ "FACE",
    INJ_BODY_PART ==  "ARM,NEC"~ "ARM",
    INJ_BODY_PART ==   "HEAD,NEC"~ "HEAD",
    INJ_BODY_PART ==  "LEG, NEC"~ "LEG",
    INJ_BODY_PART ==  "TRUNK,NEC"~ "TRUNK",
    INJ_BODY_PART ==  "BODY PARTS, NEC"  ~ "BODY",
    INJ_BODY_PART ==  "LOWER EXTREMITIES,NEC" ~ "UPEXTREMITY",
    INJ_BODY_PART ==  "UPPER EXTREMITIES, NEC"~ "LWEXTREMITY",
    
    TRUE ~ "UNKNOWN"
  )
  )

#create training example for fastText: concatenate label and text
text <- data %>% mutate(text = paste(paste("__label__", INJ_BODY_PART, sep=""), NARRATIVE))
file <- file("fasttext.train.txt")
writeLines(text$text, file)
close(file)

#use all MSHA data as training data
#use both training and pretrainedVectors [picked on from https://fasttext.cc/docs/en/english-vectors.html]
list_params = list(command = 'supervised',
                   lr = 0.1,
                   dim = 300,
                   input = file.path(getwd(), "fasttext.train.txt"),
                   output = file.path(getwd(), "fasttext.crawl.model"),
                   #output = file.path(getwd(), "fasttext.model"),
                   verbose = 2,
                   #pretrainedVectors = file.path(getwd(), "wiki-news-300d-1M-subword.vec"),
                   pretrainedVectors = file.path(getwd(), "crawl-300d-2M-subword.vec"),
                   thread = 1)

res = fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.logs_supervise.txt'),
                         MilliSecs = 5)

#once the model is learned, you will find "fasttext.model.bin" in your getwd() folder
#then use the model to predict

#####################################################################
#use the model to classify company injury description

company <- read.csv("incidents.translated.csv", encoding="latin1")
data <- unlist(company %>% filter(IncidentType=="Injury Reporting") %>% select(IncidentDescription) )
#data <- company$IncidentDescription
data <- gsub("[\r\n]", " ", data)
file<-file("fasttext.test.incidents.txt")
writeLines(data, file, sep="\n")
close(file)

#predict
list_params = list(command = 'predict-prob',
                   model = file.path(getwd(), 'fasttext.model.bin'),
                   #model = file.path(getwd(), 'fasttext.crawl.model.bin'),
                   test_data = file.path(getwd(), 'fasttext.test.incidents.txt'),
                   k = 1, 
                   th = 0.0) #only keep prediction with 0.8 probablity. See also the IncidentType (Injury Reporting)
res = fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.predict.incidents_valid.txt'))

#combine prediction with description data in one csv file
file<-file("fasttext.predict.incidents_valid.txt")
preds<-readLines(file)
result <-cbind(data, preds)
colnames(result) <- c("text","prediction")
write.csv(result, file="fasttext.predicts.incidents.csv")
result <-data.frame(result)
result <-result %>%
  mutate(part = gsub("[0-9. ]", "", gsub("__label__", "", prediction)))

injuries <- result %>%
  filter(part %in% c("BACK", "EYE", "FINGER", "WRIST", "ANKLE", "KNEE", "SHOULDER", "HAND"))


###################################################### prepare for association rule mining ###
#option 1: use document term matrix from comapny incident description 
##############################################################################################

#create document x term matrix with 703 documents
inj.corpus = tm::Corpus(VectorSource(injuries$text))
inj.corpus = tm_map(inj.corpus, content_transformer(tolower))
inj.corpus = tm_map(inj.corpus, removeNumbers)
inj.corpus = tm_map(inj.corpus, removePunctuation)
extra.stopwords = c("ee", "ees", "employee", "employees", "reported",
                    "doctor", "dr", "attempting", "trying", "stated", 
                    "started", "work", "felt", "left", "sustained", 
                    "mine", "miner", "required", "requiring", "approximately", 
                    "found", "msha", "caused", "causing", "operating", 
                    "approx", "using", "got", "getting", "onto", "went", "one",
                    "another", "wroking", "anyone")
inj.corpus = tm_map(inj.corpus, removeWords, c(extra.stopwords, stopwords::stopwords("english")))
inj.corpus =  tm_map(inj.corpus, stripWhitespace)
inspect(inj.corpus[1])

dtm_tfidf <- DocumentTermMatrix(inj.corpus, control = list(weighting = weightTfIdf))
dtm_tfidf #Sparsity 99%
#2667 terms

inspect(dtm_tfidf[1,1:20])

dtm = removeSparseTerms(dtm_tfidf, 0.99) #A numeric for the maximal allowed sparsity, adjustable
#0.99 only keep 273 terms
#0.98  127 terms
#0.95  30 terms
#0.90 7 terms

#obtain the terms that are remain
tfidf.words <- dtm$dimnames$Terms

###########
#keep word order when list words for a document
trans<-list()
for(r in 1:nrow(injuries)){
  l <- c()
  i=1
  words <-unlist(str_split(str_replace_all(injuries[r, 1], "[[:punct:]]", " "), "\\s+"))
  for(s in 1:length(words)){
    if(words[s] %in% tfidf.words){
      l[i] <- words[s]
      i = i+1
    }
  }
  
  l[i] <- paste(injuries[r, "part"], "_INJ", sep="")
  trans[[r]] <- l
}
trans

#association rule mining from trans

rules <- apriori(trans, parameter = list(support=0.01, confidence=0.8, target="rules", minlen=3)) #support and confidence adjustable
#rules.max <- subset(rules, subset=is.maximal(rules))

#quality(rules) <- cbind(quality(rules),
#                        kulc = interestMeasure(rules, measure = "kulczynski",
#                                               transactions = trans),
#                        imbalance = interestMeasure(rules, measure ="imbalance",
#                                                    transactions = trans))
#kulc and imbalance are not good quality indicators in our case, because there are multiple ways to cause the same kind of injury. 
#summary(rules)

#arules::inspect(head(rules, by="conf", n=31))

(rules <- subset(rules, subset=rhs %pin% c("INJ")))
arules::inspect(head(rules, by="conf", n=170)) #good rules have confidence close to 1
#some good rules:
#[8]   {ankle,rolled}          => {ANKLE_INJ}    0.01137980 1.0000000  0.01137980 19.527778  8 
#something rolled caused ankle injury?
#[25]  {finger,piece}          => {FINGER_INJ}   0.01422475 1.0000000  0.01422475  3.429268 10   
#[34]  {cut,finger}            => {FINGER_INJ}   0.01422475 1.0000000  0.01422475  3.429268 10  
#[60]  {finger,hit}            => {FINGER_INJ}   0.01564723 1.0000000  0.01564723  3.429268 11 
#[1]   {finger,smashed}        => {FINGER_INJ}   0.01137980 1.0000000  0.01137980  3.429268  8   
#[9]   {finger,pinched}        => {FINGER_INJ}   0.01280228 1.0000000  0.01280228  3.429268  9   
#different ways fingers can be injuried 
#[20]  {dust,eye}              => {EYE_INJ}      0.02844950 1.0000000  0.02844950  5.169118 20  
#[10]  {glasses,safety}        => {EYE_INJ}      0.01564723 1.0000000  0.01564723  5.169118 11 
#[36]  {glasses,wearing}       => {EYE_INJ}      0.01564723 1.0000000  0.01564723  5.169118 11  
#many eye_inj happened when miner was wearing safety glasses. What can we do to improve saftey glasses?
#
#some interesting rules: do back injury often happen in the morning?
#[5]   {back,morning}          => {BACK_INJ}     0.01137980 1.0000000  0.01137980  5.246269  8 
###################################################### prepare for association rule mining ###
#option 2: use topic words extracted from MSHA
##############################################################################################
#extract topic topic.words from 'text'
topics.8 <- readRDS(file="stm.topics.8.no.stem.rds")
#explore models
plot(topics.8, type="summary", xlim=c(0, 1))

words.all <- labelTopics(topics.8, n=500, seq(1:8)) #500 adjustable

#union of all top words (from prob, life, frex, and score lists) for each topic 
bag <- rbind(words.all$prob, words.all$frex, words.all$lift, words.all$score)
topic.words <-unique(c(bag))


###########
#keep word order when list words for a document
trans<-list()
for(r in 1:nrow(injuries)){
  l <- c()
  i=1
  
  words <-unlist(str_split(str_replace_all(injuries[r, 1], "[[:punct:]]", " "), "\\s+"))
  for(s in 1:length(words)){
    if(words[s] %in% topic.words){
        l[i] <- words[s]
        i = i+1
    }
  }
  
  l[i] <- paste(injuries[r, "part"], "_INJ", sep="")
  trans[[r]] <- l
}
trans


#association rule mining from trans
rules <- apriori(trans, parameter = list(support=0.005, confidence=0.6, target="rules", minlen=3)) #support and confidence adjustable
#rules.max <- subset(rules, subset=is.maximal(rules))

#quality(rules) <- cbind(quality(rules),
#                        kulc = interestMeasure(rules, measure = "kulczynski",
#                                               transactions = trans),
#                        imbalance = interestMeasure(rules, measure ="imbalance",
#                                                    transactions = trans))
#summary(rules)

arules::inspect(head(rules, by="conf", n=31))

(rules <- subset(rules, subset=rhs %pin% c("INJ")))
arules::inspect(head(rules, by="conf", n=170)) #good rules have confidenct close to 1


#############################################################################
#use the model learned from MSHA data to predict injury from interaction data
#use interaction/audit data as test
interaction <- read.csv("SafetyInteractions.csv", encoding="latin1")

interaction <- interaction %>% 
                mutate(selected = case_when (
                  startsWith(field4, "Interaction") ~1,
                  startsWith(field4, "Audit") ~1,
                  TRUE~0
                  )
                )%>%filter(selected==1)
                
file <- file("fasttext.test.txt")
writeLines(interaction[,4], file)
close(file)

#predict
list_params = list(command = 'predict-prob',
                   model = file.path(getwd(), 'fasttext.model.bin'),
                   test_data = file.path(getwd(), 'fasttext.test.txt'),
                   k = 3,
                   th = 0.0)
res = fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.predict_valid.txt'))

#combine prediction with interaction/audit data in one csv file
file<-file("fasttext.predict_valid.txt")
preds<-readLines(file)
result <-cbind(interaction$field4, preds)
write.csv(result, file="fasttext.predicts.csv")



