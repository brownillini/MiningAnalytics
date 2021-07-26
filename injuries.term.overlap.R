#check vocabulary overlap within and among different injuries 

#examine the output in the end, we see that the words used to describe different types of injuries have 30% or more in common. 
#One can also see words used for HAND injuries are almost covered by words used for BACK injuries
#This explains why it is difficult to create "clear" topic models that corresponding to injured body parts.
library(dplyr)
library(stopwords)


setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1") #company data does not contain injured body parts label, so we use MSHA data.

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


extra.stopwords <- c("ee", "ees", "employee", "employees", "reported",
                     "doctor", "dr", "attempting", "trying", "stated", 
                     "started", "work", "felt", "left", "sustained", 
                     "mine", "miner", "required", "requiring", "approximately", 
                     "found", "msha", "caused", "causing", "operating", 
                     "approx", "using", "got", "getting", "onto", "went", "one",
                     "another", "wroking", "anyone")
all.stopwords <- c(extra.stopwords, stopwords::stopwords("english"))

type <- c("EYE", "HAND", "FINGER", "WRIST", "ANKLE", "KNEE", "SHOULDER", "BACK")
eye <- c();
hand <-c();
finger<- c();
wrist <- c();
ankle <- c();
knee <- c();
shoulder <-c();
back <-c();
list <-list(eye, hand, finger, wrist, ankle, knee, shoulder, back)

'%nin%' <- Negate('%in%')
for(t in 1:length(type)){
  words <-unlist(str_split(str_replace_all(tolower(data[data$INJ_BODY_PART==type[t], 1]), "[[:punct:]]", " "), "\\s+"))
  list[[t]] <- words[words %nin% all.stopwords]
}

#pair-wise comparison of words in list
for(t1 in 1:(length(list)-1)){
  for(t2 in (t1+1):length(list)){
    total <- length(union(list[[t1]], list[[t2]]))
    intersection <- length(intersect(list[[t1]], list[[t2]]))
    diff.ab<- length(setdiff(list[[t1]], list[[t2]]))
    diff.ba<- length(setdiff(list[[t2]], list[[t1]]))
    print(paste(type[t1], "vs", type[t2], ": total =", total, "; intersection =", 
                round(intersection/total, 2), "; diff.a.b =", round(diff.ab/total, 2), 
                "; diff.b.a =", round(diff.ba/total, 2)))
  }
}


