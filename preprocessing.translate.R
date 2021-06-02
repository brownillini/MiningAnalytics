install.packages("textreuse") #description similarity
install.packages("googleLanguageR")
install.packages("cld2")

library(textreuse) #https://cran.r-project.org/web/packages/textreuse/index.html
library(googleLanguageR)
library(cld2)
library(tidyverse)

setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw.actions <- read.csv("Corrective Actions List - COMBINED.csv")
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





