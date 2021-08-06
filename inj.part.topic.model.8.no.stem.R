#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)

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
processed <- textProcessor( #TRUE/FALSE all adjustable
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
                      "approx", "using", "got", "getting", "onto", "went", "one", "another", "wroking", "anyone"), #adjustable
  custompunctuation = NULL,
  v1 = FALSE
)

#View(stopwords(language = "en",source = "smart"))
#data$NARRATIVE <- gsub("x-ray", "xray", data$NARRATIVE)
#data$NARRATIVE <- gsub("X-ray", "xray", data$NARRATIVE)
#sum(str_count(data$NARRATIVE, "xray"))/nrow(data)
#sum(str_count(data$NARRATIVE, "back"))/nrow(data) # 36855
#sum(str_count(data$NARRATIVE, "finger"))/nrow(data) #25162 or 0.104
#sum(str_count(data$NARRATIVE, "ankle"))/nrow(data) #7458  or 0.03
#nrow(data) #88501 + 30000
#plotRemoved(processed$documents, lower.thresh = seq(1, 500, by = 100))
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, 
                     lower.thresh = 2, upper.thresh = 8000)

#Content (words->topic) vs. Prevalence (document -> topic)
#Topical prevalence refers to how much of a document is associated with a topic 
#and topical content refers to the words used within a topic.

stm.topics.8 <- stm(documents = out$documents, vocab = out$vocab,
                    K = 8, prevalence =~ INJ_BODY_PART,
                    data = out$meta,
                    init.type = "Spectral")
saveRDS(stm.topics.8, file="stm.topics.8.no.stem.rds")

topics <- readRDS(file="stm.topics.8.no.stem.rds")
#top 20 words of the 8 topics
labelTopics(topics, n=20, seq(1:8))
#explore models
plot(topics, type="summary", xlim=c(0, 1))

