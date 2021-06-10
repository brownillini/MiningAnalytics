#explore data related to contractors
library(tidyverse)
setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R/MiningAnalytics")
incidents <- read.csv("incidents.translated.preprocessed.csv", na.strings=c("", "NA"))
msha <- read.csv("MSHA.injuries.full.csv", na.strings=c("", "NA"))

injuries <- msha %>% filter(MINE_ID %in% c("1000088", "2602314", "2602691", "5001267"))
nrow(injuries) #1018

sort(table(injuries$CAL_YR)) #from 2000 to 2021

injuries.2019 <- msha %>% 
  filter(MINE_ID %in% c("1000088", "2602314", "2602691", "5001267")) %>% 
  filter(CAL_YR == 2019)
nrow(injuries.2019) #17

incidents.2019 <- incidents %>% filter(YearOfOccurance == 2019)
nrow(incidents.2019) #1457

#nrow(join)=0, so no descriptions in two datasets are the same
join <- inner_join(injuries.2019, incidents.2019, by = c("NARRATIVE" = "IncidentDescription"))

#find 'contractor' in msha
injuries.contractor <- msha %>% filter(!is.na(CONTRACTOR_ID))  
injuries.contractor <- msha[-c(grep(NA, msha$CONTRACTOR_ID)),] 
injuries.narrative.contract <- msha[grep("contract", msha$NARRATIVE), ]
nrow(injuries.contractor) #24893
nrow(injuries.narrative.contract) #691

#find 'contractor' in company
incidents.narrative.contract <- incidents[grep("contract", incidents$IncidentDescription), ]
nrow(incidents.narrative.contract) #26

incidents.location.contract <- incidents[grep("contract", incidents$Location), ]
nrow(incidents.location.contract) #240

#total
incidents.contract.total<- incidents[union(grep("contract", incidents$Location),
                                           grep("contract", incidents$IncidentDescription)),]
nrow(incidents.contract.total) #265
