#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)

install.packages("fastText")
library(fastText)
library(dplyr)

#use MSHA inj_body_part data to train a model using fastText
#use the model to predict which body part injury an interaction/audit could be a leading factor

#https://towardsdatascience.com/fasttext-bag-of-tricks-for-efficient-text-classification-513ba9e302e7
#https://fasttext.cc/docs/en/supervised-tutorial.html
#using an example data set: data
setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1")

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
                   output = file.path(getwd(), "fasttext.model"),
                   verbose = 2,
                   pretrainedVectors = file.path(getwd(), "wiki-news-300d-1M-subword.vec"),
                   thread = 1)

res = fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.logs_supervise.txt'),
                         MilliSecs = 5)

#once the model is learned, you will find "fasttext.model.bin" in your getwd() folder
#then use the model to predict

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



