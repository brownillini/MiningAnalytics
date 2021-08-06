#!/usr/bin/env Rscript
args = commandArgs(trailingOnly = TRUE)

#best number of topics
#use stm to build topic models and explore the models
library(tidyverse)
library(stm)
library(stopwords)

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

data <-data %>% filter (INJ_BODY_PART %in% c("EYE", "HAND", "FINGER", "WRIST", "ANKLE", "KNEE", "SHOULDER", "BACK"))

data$NARRATIVE <- gsub("-", "_", data$NARRATIVE)

#https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf
processed <- textProcessor(
  data$NARRATIVE,
  metadata = data,
  lowercase = TRUE,
  removestopwords = TRUE,
  removenumbers = TRUE,
  removepunctuation = TRUE,
  ucp = FALSE,
  stem =FALSE,
  wordLengths = c(3, Inf),
  sparselevel = 1,
  language = c("en"),
  verbose = TRUE,
  onlycharacter = FALSE,
  striphtml = FALSE,
  customstopwords = c("ee", "ees", "employee", "employees", "reported",
                      "doctor", "dr", "attempting", "trying", "stated", 
                      "started", "work", "felt", "left", "sustained", 
                      "mine", "miner", "required", "requiring", "approximately", 
                      "found", "msha", "caused", "causing", "operating", 
                      "approx", "using", "got", "getting", "onto", "went", "one",
                      "another", "wroking", "anyone"), #adjustable
  custompunctuation = NULL,
  v1 = FALSE
)


#plotRemoved(processed$documents, lower.thresh = seq(1, 500, by = 100))
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, 
                     lower.thresh = 5, upper.thresh = 30000) #both numbers are adjustable
#total documents:129977. total unique word: 36882
#.t=2 u.t = 8000: 
#Removing 25199 of 36882 terms (316633 of 1817604 tokens) due to frequency 
#Removing 41 Documents with No Words 
#Your corpus now has 129936 documents, 11683 terms and 1500971 tokens.

#l.t=2 u.t = 30000
#Removing 25182 of 36882 terms (91776 of 1817604 tokens) due to frequency 
#Your corpus now has 129977 documents, 11700 terms and 1725828 tokens.

#l.t=5 u.t = 30000
#Removing 29067 of 36882 terms (106190 of 1817604 tokens) due to frequency 
#Your corpus now has 129977 documents, 7815 terms and 1711414 tokens

#l.t=5 u.t = 8000
#Removing 29084 of 36882 terms (331047 of 1817604 tokens) due to frequency 
#Removing 43 Documents with No Words. these documents contains only high frequency terms 8000-30000 occurance
#Your corpus now has 129934 documents, 7798 terms and 1486557 tokens

topics.Ks <- searchK(out$documents, out$vocab, K=seq(2, 50), prevalence =~ INJ_BODY_PART, data = out$meta)

saveRDS(topics.Ks, file="topics.Ks.RDS")

topics.Ks$results


#stm.topics.8 <- stm(documents = out$documents, vocab = out$vocab,
#                    K = 8, prevalence =~ INJ_BODY_PART,
#                    data = out$meta,
#                    init.type = "Spectral")
#saveRDS(stm.topics.8, file="stm.topics.8.no.stem.rds")
