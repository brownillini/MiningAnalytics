#use stm to build topic models and explore the models
install.packages("rsvd")

library(tidyverse)
library(stm)
library(stopwords)
library(gridExtra)
#replace textprocessor function in stm
library(quanteda) #dfm a document term matrix that can be supplied directly to the stm model fitting function


setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1")

data <- raw1%>%select(NARRATIVE, NATURE_INJURY, INJ_BODY_PART) %>%filter(NATURE_INJURY %in% c("DISLOCATION", "POISONING,SYSTEMIC"))



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
  customstopwords = c("ee", "ees", "employee", "employees", "reported",
                      "doctor", "dr", "attempting", "trying", "stated", 
                      "started", "work", "felt", "left", "sustained", 
                      "mine", "required", "requiring", "approximately", 
                      "found", "msha", "caused", "causing", "operating", 
                      "approx"),
  custompunctuation = NULL,
  v1 = FALSE
)

View(stopwords(language = "en",source = "smart"))
data$NARRATIVE <- gsub("x-ray", "xray", data$NARRATIVE)
data$NARRATIVE <- gsub("X-ray", "xray", data$NARRATIVE)
sum(str_count(data$NARRATIVE, "xray"))/nrow(data)
sum(str_count(data$NARRATIVE, "back"))/nrow(data) # 36855
sum(str_count(data$NARRATIVE, "finger"))/nrow(data) #25162 or 0.104
sum(str_count(data$NARRATIVE, "ankel"))/nrow(data) #9  or 3.724426e-05

plotRemoved(processed$documents, lower.thresh = seq(1, 500, by = 100))
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, 
                     lower.thresh = 2, upper.thresh = 50000 )

#Content (words->topic) vs. Prevalence (document -> topic)
#Topical prevalence refers to how much of a document is associated with a topic 
#and topical content refers to the words used within a topic.

topics.2 <- stm(documents = out$documents, vocab = out$vocab,
                           K = 2, prevalence =~ NATURE_INJURY, content =~ NATURE_INJURY,
                           data = out$meta,
                           init.type = "Spectral")
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
#explore models
plot(topics.2, type="summary", xlim=c(0, 1))
plot(topics.2.selected, type="summary", xlim=c(0, 1))

#words-topics
labelTopics(topics.2, n=20, c(1,2))

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

plot.STM(topics.2)

#documents-topics
narratives <- findThoughts(topics.2, texts=out$meta$NARRATIVE, n=2, topics=1)$docs[[1]]
par(mfrow = c(1, 1),mar = c(0, 0, 0, 0))
plotQuote(narratives, width = 60, main = "Topic 1")

#estimate metadata/topic relationships
out$meta$NATURE_INJURY <-as.factor(out$meta$NATURE_INJURY)
effect <- estimateEffect(1:2 ~ NATURE_INJURY, topics.2, meta=out$meta, uncertainty = "Global")
summary(effect, topics=1)

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
(mod.out.corr <- topicCorr(topics.2))
plot(mod.out.corr)
