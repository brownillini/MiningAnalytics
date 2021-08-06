#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)

#use stm to build topic models and explore the models
install.packages("rsvd")
install.packages("stm")

library(tidyverse)
library(stm)
library(stopwords)
library(gridExtra)
#replace textprocessor function in stm
#library(quanteda) #dfm a document term matrix that can be supplied directly to the stm model fitting function


setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
#raw <- read.csv("MSHA.injuries.full.csv", encoding="latin1")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1")

#focus on 
#1.	Eye injuries
#2.	Hand injuries 
#3.	Sprains and strains of the ankle, knee, shoulder or back
#4.	Fractures and amputations
#5.	Fatal injuries
#6.	Heat stress 
#7.	Noise dose
#8.	Diesel particulate matter (DPM) exposure
#9.	Fatigue
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
    TRUE ~ "OTHER"
  )
)
other <-data[data$INJ_BODY_PART=="OTHER", ]
set.seed(1234)
other <-sample_n(other, 30000)
data <-data %>% filter (INJ_BODY_PART %in% c("EYE", "HAND", "ANKLE","FINGER", "WRIST", "KNEE", "SHOULDER", "BACK"))
data <- rbind(data, other)

#saveRDS(data, file="fastText.data.RDS")
data$NARRATIVE <- gsub("x-ray", "xray", data$NARRATIVE)
data$NARRATIVE <- gsub("X-ray", "xray", data$NARRATIVE)

#https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf
processed <- textProcessor(
  data$NARRATIVE,
  metadata = data,
  lowercase = TRUE,
  removestopwords = TRUE,
  removenumbers = TRUE,
  removepunctuation = TRUE,
  ucp = FALSE,
  stem =TRUE,
  wordLengths = c(3, Inf),
  sparselevel = 1,
  language = c("en"),
  verbose = TRUE,
  onlycharacter = FALSE,
  striphtml = FALSE,
  customstopwords = c("approx","approximately","attempting","caused", "causing","doctor","dr",
                      "ee", "ees", "employee", "employees", 
                      "felt","found","got", "left", "mine","msha",
                      "operating","reported","required", "requiring", 
                      "started","stated","sustained",  "trying","work"
                      ), #adjust
  custompunctuation = NULL,
  v1 = FALSE
)

#View(stopwords(language = "en",source = "smart"))
#sum(str_count(data$NARRATIVE, "xray"))/nrow(data)
#sum(str_count(data$NARRATIVE, "back"))/nrow(data) # 36855
#sum(str_count(data$NARRATIVE, "finger"))/nrow(data) #25162 or 0.104
#sum(str_count(data$NARRATIVE, "ankle"))/nrow(data) #7458  or 0.03
#nrow(data) #88501 + 30000
#plotRemoved(processed$documents, lower.thresh = seq(1, 500, by = 100))
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, 
                     lower.thresh = 2, upper.thresh = 30000)

#Content (words->topic) vs. Prevalence (document -> topic)
#Topical prevalence refers to how much of a document is associated with a topic 
#and topical content refers to the words used within a topic.

stm.topics.9 <- stm(documents = out$documents, vocab = out$vocab,
                           K = 9, prevalence =~ INJ_BODY_PART,
                           data = out$meta,
                           init.type = "Spectral")
saveRDS(stm.topics.9, file="stm.topics.9.rds")

########################################
topics.9 <- readRDS(file="stm.topics.9.rds")
#top 20 words of the 9 topics
labelTopics(topics.9, n=20, seq(1:9))

#highest probability words
#FREX weights words by their overall frequency and how exclusive they are to the topic
#Lift weights words by dividing by their frequency in other topics, therefore giving higher weight to words that appear less frequently in other topics.
#score divides the log frequency of the word in the topic by the log frequency of the word in other topics.

topics.7 <- readRDS(file="stm.topics.7.rds")
#top 20 words of the 7 topics
labelTopics(topics.7, n=20, seq(1:7))


#explore models
plot(topics.9, type="summary", xlim=c(0, 1))
#or
plot.STM(topics.9)



#Topic 1 Top Words:
#Highest Prob: disloc, shoulder, right, fell, hand, slip, finger 
#FREX: disloc, shoulder, fell, slip, finger, step, fall 
#Lift: backward, beam, bottom, concret, fall, hydraul, ice 
#Score: disloc, shoulder, fell, slip, fall, step, ground 
#Topic 2 Top Words:
#Highest Prob: stung, bee, bite, spider, medic, day, hospit 
#FREX: stung, bee, bite, spider, treatment, bitten, sting 
#Lift: accumul, addit, admiss, advers, alcoa, although, ant 
#Score: stung, bee, spider, bite, fume, inhal, breath 



#documents-topics
narratives <- findThoughts(topics.9, texts=out$meta$NARRATIVE, n=2, topics=1)$docs[[1]]
par(mfrow = c(1, 1),mar = c(0, 0, 0, 0))
plotQuote(narratives, width = 60, main = "Topic 1")

#estimate metadata/topic relationships
#out$meta$NATURE_INJURY <-as.factor(out$meta$NATURE_INJURY)
#effect <- estimateEffect(1:2 ~ NATURE_INJURY, topics.2, meta=out$meta, uncertainty = "Global")
#summary(effect, topics=1)

out$meta$INJ_BODY_PART <-as.factor(out$meta$INJ_BODY_PART)
effect <- estimateEffect(c(2) ~ INJ_BODY_PART, topics.9, meta=out$meta, uncertainty = "Global")
summary(effect, topics=2)

png("effect2.png", width=980, height=980)
plot.estimateEffect(effect, "INJ_BODY_PART", model=gadarianFit, method="pointestimate")
dev.off()
plot(topics.9, type = "perspectives", topics = c(1, 3))
#Topic 1 = EYE
#2 =
#3 =
#4 = FINGER/HAND
#5 =


################################
#visualization
#wordcloud
cloud(topics.2, topic=1, type="model", max.words = 50)
cloud(topics.2, topic=2, type="model", max.words = 50)
cloud(topics.2, topic=2, type="documents", max.words = 50, thresh=0.9, documents=out$documents)

cloud(topics.2.selected)
#plot: these two are the same
plot.STM(topics.2)
plot(topics.2, type="summary", xlim=c(0, 1))
plot(topics.2, type="perspectives", topics=c(1,2), xlim=c(0, 1))


#plot effect of covariate of the topics
#method="continous" is useful for continuous content covariate
plot(effect, covariate = "NATURE_INJURY", topics = c(1, 2),
        model = topics.2, method = "difference",
        cov.value1 = "DISLOCATION", cov.value2 = "POISONING,SYSTEMIC",
         xlab = "More poison ... More dislocation",
         main = "Effect of Dislocation vs. Poison",
         xlim = c(-1, 1), labeltype = "custom",
         custom.labels = c('disloc', 'stung'))

#plot covariate interaction #not working yet!!

#topic correlation
(mod.out.corr <- topicCorr(topics.9, cutoff = 0.1))
plot(mod.out.corr)


###################################
#when not using 'Spectral', one can use different init methods to produce multiple models
topics.select <- selectModel(documents = out$documents, vocab = out$vocab,
                             K = 2, prevalence =~ NATURE_INJURY,
                             data = out$meta,
                             runs = 10, seed=1324576) #10
topics.select$runout #2 are selected out of 10
par(mfrow=c(1,1))
plotModels(topics.select, labels = 1:length(topics.select$runout), pch=c(1,2), legend.position="bottomright")
#models 1 and 2 very similar

topics.2.selected <- topics.select$runout[[1]]
#################################
#try out different Ks
topics.Ks <- searchK(out$documents, out$vocab, K=c(2, 3, 4), prevalence =~ NATURE_INJURY, data = out$meta)
topics.Ks$results


library(rsvd)
#let the algorithm to find K, result is non-deterministic, K can be different with each different seed
topics.fount.K <- searchK(out$documents, out$vocab, K=0, prevalence =~ NATURE_INJURY, data = out$meta, init.type = "Spectral")
#61 topics. Topic # not reported in the results but given though verbose output of the algorithm.
################################
#####
#TODO:
#leading vs. lagging
#predict potential injury using interaction data (interaction holds the potential leading factors)

