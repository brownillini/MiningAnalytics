#!/usr/bin/env Rscript

args = commandArgs(trailingOnly = TRUE)


library(fastText)
#library(stm)
library(dplyr)
library(stringr)
#library(tm)
#library(stopwords)

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
#adjustable: labels after ~ are adjustable.
data <- raw1 %>% select(NARRATIVE, INJ_BODY_PART) %>% 
  mutate(INJ_BODY_PART = case_when (
    INJ_BODY_PART == "EYE(S) OPTIC NERVE/VISON" ~ "EYE",
    INJ_BODY_PART == "HAND (NOT WRIST OR FINGERS)" ~ "HAND",
    INJ_BODY_PART == "FINGER(S)/THUMB" ~ "HAND",
    INJ_BODY_PART == "WRIST" ~ "HAND",
    INJ_BODY_PART == "ANKLE" ~  "ANKLE",
    INJ_BODY_PART == "KNEE/PATELLA" ~ "KNEE",
    INJ_BODY_PART == "SHOULDERS (COLLARBONE/CLAVICLE/SCAPULA)" ~ "SHOULDER",
    INJ_BODY_PART == "BACK (MUSCLES/SPINE/S-CORD/TAILBONE)" ~ "BACK",
    INJ_BODY_PART == "FOREARM/ULNAR/RADIUS" ~ "OTHER",
    INJ_BODY_PART ==  "ABDOMEN/INTERNAL ORGANS"~ "OTHER",
    INJ_BODY_PART ==  "HIPS (PELVIS/ORGANS/KIDNEYS/BUTTOCKS)"~ "OTHER",
    INJ_BODY_PART ==  "ELBOW" ~ "OTHER",
    INJ_BODY_PART ==  "FOOT(NOT ANKLE/TOE)/TARSUS/METATARSUS"~ "OTHER",
    INJ_BODY_PART ==  "MOUTH/LIP/TEETH/TONGUE/THROAT/TASTE"~ "OTHER",
    INJ_BODY_PART ==  "SCALP" ~ "OTHER",
    INJ_BODY_PART ==  "CHEST (RIBS/BREAST BONE/CHEST ORGNS)"~ "OTHER",
    INJ_BODY_PART ==  "LOWER LEG/TIBIA/FIBULA"~ "OTHER",
    INJ_BODY_PART ==  "NECK"~ "OTHER",
    INJ_BODY_PART ==  "JAW INCLUDE CHIN" ~ "OTHER",
    INJ_BODY_PART ==  "TOE(S)/PHALANGES" ~ "OTHER",
    INJ_BODY_PART ==  "EAR(S) INTERNAL & HEARING" ~ "OTHER",
    INJ_BODY_PART ==  "UPPER ARM/HUMERUS"~ "OTHER",
    INJ_BODY_PART ==  "BRAIN" ~ "OTHER",
    INJ_BODY_PART ==  "THIGH/FEMUR"  ~ "OTHER",
    INJ_BODY_PART ==  "NOSE/NASAL PASSAGES/SINUS/SMELL"  ~ "OTHER",
    INJ_BODY_PART ==  "EAR(S) EXTERNAL"~ "OTHER",
    INJ_BODY_PART ==  "SKULL"~ "OTHER",
    INJ_BODY_PART ==  "EAR(S) INTERNAL & EXTERNAL" ~ "OTHER",

    INJ_BODY_PART ==  "BODY SYSTEMS"~ "EXCLUDE",
    INJ_BODY_PART ==  "MULTIPLE PARTS (MORE THAN ONE MAJOR)"~ "EXCLUDE",
    INJ_BODY_PART ==  "TRUNK, MULTIPLE PARTS" ~ "EXCLUDE",
    INJ_BODY_PART ==  "UPPER EXTREMITIES, MULTIPLE"~ "EXCLUDE",
    INJ_BODY_PART ==  "LOWER EXTREMITIES, MULTIPLE PARTS"~ "EXCLUDE",
    INJ_BODY_PART ==  "FACE, MULTIPLE PARTS" ~ "EXCLUDE",
    INJ_BODY_PART ==  "ARM, MULTIPLE PARTS" ~ "EXCLUDE",
    INJ_BODY_PART ==  "HEAD, MULTIPLE PARTS"~ "EXCLUDE",
    INJ_BODY_PART ==  "LEG, MULTIPLE PARTS" ~ "EXCLUDE",
    
    INJ_BODY_PART ==  "FACE,NEC" ~ "OTHER",
    INJ_BODY_PART ==  "ARM,NEC"~ "OTHER",
    INJ_BODY_PART ==   "HEAD,NEC"~ "OTHER",
    INJ_BODY_PART ==  "LEG, NEC"~ "OTHER",
    INJ_BODY_PART ==  "TRUNK,NEC"~ "OTHER",
    INJ_BODY_PART ==  "BODY PARTS, NEC"  ~ "OTHER",
    INJ_BODY_PART ==  "LOWER EXTREMITIES,NEC" ~ "OTHER",
    INJ_BODY_PART ==  "UPPER EXTREMITIES, NEC"~ "OTHER",
    
    TRUE ~ "UNKNOWN"
  )
  )

other <-data[data$INJ_BODY_PART=="OTHER", ]
set.seed(1234)
other <-sample_n(other, 30000)

data <- data %>%
  filter(INJ_BODY_PART %in% c("BACK", "EYE", "ANKLE", "KNEE", "SHOULDER", "HAND"))
data <- rbind(data, other)

#create training example for fastText: concatenate label and text
text <- data %>% mutate(text = paste(paste("__label__", INJ_BODY_PART, sep=""), NARRATIVE))
file <- file("fasttext.train.inter.txt")
writeLines(text$text, file)
close(file)

#learn the model on UA HPC
#use all MSHA data as training data
#use both training and pretrainedVectors [picked one from https://fasttext.cc/docs/en/english-vectors.html]
list_params = list(command = 'supervised',
                   lr = 0.1,
                   dim = 300,
                   input = file.path(getwd(), "fasttext.train.inter.txt"),
                   output = file.path(getwd(), "fasttext.inter.model"),
                   verbose = 2,
                   #pretrainedVectors = file.path(getwd(), "wiki-news-300d-1M-subword.vec"),
                   pretrainedVectors = file.path(getwd(), "crawl-300d-2M-subword.vec"),
                   thread = 1)

res = fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.inter.logs_supervise.txt'),
                         MilliSecs = 5)



############################# IMPORTANT IMPORTANT IMPORTANT #############################################
#NOTE: I had to run the supervised training part on UA HPC to produce the model fasttext.inter.model
#but when use the model in the code that follows on my windows laptop, it doesn't produce any prediction.
#I had to run the code that follows on UA HPC to get the test result
#############################################################################

#############################################################################
#use the model learned from MSHA data to predict injury from interaction data
#use labeled interaction/audit data as test
#create validation set from human labeled data

labeled <- read.csv("interaction.labeled.csv", encoding="latin1")

#valid <- labeled  %>% group_by_all() %>% mutate(label = paste(union(unlist(str_split(Label..R.Reed., "\\.", simplify = TRUE )),
#                                                 unlist(str_split(Label..L.Brown., "\\.", simplify = TRUE))), sep = '', collapse= " __label__")) %>% select(DESC, label)

valid <- labeled %>% group_by_all()  %>% mutate(label = paste("__label__", paste(union(c(str_split(str_trim(Label..R.Reed.), "[.]", simplify = TRUE )),
                                                                    c(str_split(str_trim(Label..L.Brown.), "[.]", simplify = TRUE))), sep = '', collapse= " __label__"), sep=''))

text <- valid %>% ungroup() %>% mutate(text = paste(label, DESC)) %>% select(text)

file <- file("fasttext.inter.test.txt")
writeLines(text$text, file)
close(file)

#test, RUN THIS ON UA HPC
list_params <- list(command = 'predict',
                   model = file.path(getwd(), 'fasttext.inter.model.bin'), # don't know why this model does not predict anything
                   test_data = file.path(getwd(), 'fasttext.inter.test.txt'),
                   k = 4, #adjustable, output topic three best predictions
                   th = 0.01) #adjustable, threshold for the probability, predictions with a probability >= th will be output
res <- fasttext_interface(list_params,
                         path_output = file.path(getwd(), 'fasttext.inter.predict_valid.txt'))

#Follow Precision and Recall scores were obtained on UA HPC

#th=0
# N       500
# P@1     0.58
# R@1     0.227
# N       500
# P@2     0.535
# R@2     0.419
# N       500
# P@3     0.504
# R@3     0.592
# N       500
# P@4     0.476
# R@4     0.745


#th=0.01
# N       500
# P@1     0.58
# R@1     0.227
# N       500
# P@2     0.538
# R@2     0.416
# N       500
# P@3     0.507
# R@3     0.579
# N       500
# P@4     0.486
# R@4     0.723




#combine prediction with interaction/audit data in one csv file
#file<-file("fasttext.inter.predict_valid.txt")
#preds<-readLines(file)
#result <-cbind(text, preds)
#write.csv(result, file="fasttext.inter.predicts.csv")
#close(file)



