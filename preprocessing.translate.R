install.packages("textreuse") #description similarity
install.packages("googleLanguageR")
install.packages("cld2")
install.packages("cld3")

library(textreuse) #https://cran.r-project.org/web/packages/textreuse/index.html
library(googleLanguageR)
#library(cld2)
#library(cld3)
library(tidyverse)
library(fastText)

setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")

######################### remove French duplicates from MSHA injuries data

#tested cld2 and fasttext. results are quite bad.
#vec_txt = c("roses are red", "las rosas son rojas.", "cuanto es el banano")
#file_pretrained = system.file("language_identification/lid.176.ftz", package = "fastText")

#msha <- read.csv("MSHA.injuries.csv")
#for(i in 1:length(msha$NARRATIVE)){
#  text <- msha$NARRATIVE[i]
#  vector<- c(str_split(text, "[.?!] +(?=[A-Z(])", simplify = TRUE))
#  languages <- language_identification(input_obj = vector,
#                          pre_trained_language_model_path = file_pretrained,
#                          k = 1)[,1]
#  #languages <- cld2::detect_language(vector, plain_text = TRUE)
#  new <- ""
#  for(j in 1:length(vector)){
#    if(is.na(languages[j]) || languages[j]=="en"){
#    #if(is.na(languages[j]) || (languages[j]!="fr" && languages[j] !="es")){#not French and not Spanish
#      new <- paste(new, vector[j], sep=" ")
#    }else{
#     print(paste(languages[j], ":",vector[j]))
#    }
#  }
#  msha$NARRATIVE[i] <- new
#}


######################### translate descriptions from French to English
#this works!
#raw.actions <- read.csv("Corrective Actions List - COMBINED.csv")
raw.incidents <- read.csv("Incident Management - COMBINED.csv")


(detected_language <- sapply(raw.incidents$IncidentDescription, detect_language)) %>% data.frame(check.names = FALSE)

tr.incidents <- raw.incidents
#2240 french
for(i in 1:nrow(tr.incidents)){
#for(i in 1:5){
  if(!is.na(detected_language[[i]]) & detected_language[[i]]=="fr"){
    tr.incidents[i, 'IncidentDescription'] <- gl_translate(tr.incidents[i, 'IncidentDescription'], target="en")
  }
}

copy <- tr.incidents

tr.incidents$language <- NULL

for(i in 1:nrow(tr.incidents)){
  if(!is.na(detected_language[[i]]) & detected_language[[i]]=="fr"){
    tr.incidents[i, "language"] <-"fr"
  }else if(!is.na(detected_language[[i]]) & detected_language[[i]]=="en"){
    tr.incidents[i, "language"] <-"en"
  }
}

  
write.csv( tr.incidents, "incidents.translated.csv")

nrow(incidents)
sum(complete.cases(incidents))


sort(unique(incidents$Location)) #111 unique locations - may create concept hierarchy
plot(sort(table(incidents$Location))) #incidents not evenly distributed among locations
#Q: location data may not be at the same granularity level: LF Mill Department vs. Hecla Quebec

incidents<- incidents %>% 
  mutate(MonthOfOccurance = month(as.Date(DateofOccurrence, format="%m/%d/%Y"))) %>%
  mutate(YearOfOccurance = year(as.Date(DateofOccurrence, format="%m/%d/%Y")))

sort(table(incidents$MonthOfOccurance))
sort(table(incidents$YearOfOccurance)) #2017 data incomplete. 2021 only have the first 4 month

sort(unique(incidents$IncidentType)) #need OHE
sort(unique(incidents$Hazard))
sort(unique(incidents$NearMiss))
sort(unique(incidents$RiskRating))
sort(unique(incidents$InvolvedPerson))
sort(table(incidents$InvolvedPerson)) #some people appear more often, majority only once

sort(unique(incidents$ReportedBy))
sort(table(incidents$ReportedBy))

sort(unique(incidents$InjuredParty)) #diff from InvolvedPerson
sort(table(incidents$InjuredParty))

sort(unique(incidents$SubIncidentPersonResponsible))
sort(unique(incidents$PersonResponsible))
sort(table(incidents$language))

#incidents$IncidentType OHE:

list <-as.list(incidents)
incidents <- data.frame(cbind(list, mtabulate(strsplit(as.character(list$IncidentType), "/|\n\n"))))

write.csv(incidents, "incidents.translated.preprocessed.csv")





