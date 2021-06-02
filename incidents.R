install.packages("textreuse") #description similarity
install.packages("googleLanguageR")
install.packages("cld2")

library(textreuse)
library(googleLanguageR)
library(cld2)
library(tidyverse)

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
tr.incidents$language <- null
for(i in 1:nrow(tr.incidents)){
  if(!is.na(detected_language[[i]]) & detected_language[[i]]=="fr"){
    tr.incidents[i, "language"] <-"fr"
  }else if(!is.na(detected_language[[i]]) & detected_language[[i]]=="en"){
    tr.incidents[i, "language"] <-"en"
  }
}

  
write.csv( tr.incidents, "incidents.translated.csv")





