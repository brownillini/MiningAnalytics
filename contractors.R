#explore data related to contractors
library(tidyverse)
setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
incidents <- read.csv("incidents.translated.preprocessed.csv", na.strings=c("", "NA"))
msha <- read.csv("MSHA.injuries.full.csv", na.strings=c("", "?", "NO VALUE FOUND", "Not Reported"))

injuries.com <- msha %>% filter(MINE_ID %in% c("1000088", "2602314", "2602691", "5001267"))
nrow(injuries.com) #total company submitted injuries in msha over years = 1018

sort(table(injuries.com$CAL_YR)) #from 2000 to 2021

#company data and msha data are not the same: in 2019, more cases in company dataset
injuries.com.2019 <- msha %>% 
  filter(MINE_ID %in% c("1000088", "2602314", "2602691", "5001267")) %>% 
  filter(CAL_YR == 2019)
nrow(injuries.com.2019) #17

incidents.2019 <- incidents %>% filter(YearOfOccurance == 2019)
nrow(incidents.2019) #1457


join <- inner_join(injuries.com.2019, incidents.2019, by = c("NARRATIVE" = "IncidentDescription"))
#nrow(join)=0, so no descriptions in two datasets are the same

###########################################
#find 'contractor' in msha. 
#****************************************************************************************************
#Assuming non-NA contractor_ID and narrative containing 'contract' are injuries related to contractors
injuries.contractor <- msha %>% filter(!is.na(CONTRACTOR_ID))  
injuries.narrative.contract <- msha[grep("contract", msha$NARRATIVE, ignore.case=TRUE), ]
nrow(injuries.contractor) #24893
nrow(injuries.narrative.contract) #691
contractors.msha<- msha %>% mutate(isContractor = case_when((!is.na(CONTRACTOR_ID) | grepl("contract", NARRATIVE, ignore.case=TRUE))~1, TRUE~0)) 

## how incident profile of contractors is different from the incident profile of non-contractor incidents?

tab <- with(contractors.msha[, c("isContractor", "SUBUNIT_CD")], table(isContractor, SUBUNIT_CD))
chisq.test(tab) #sig.
prop.table(tab, margin=1)


unique(msha[, c("DEGREE_INJURY_CD", "DEGREE_INJURY")])
tab <- with(contractors.msha[, c("isContractor", "DEGREE_INJURY_CD")], table(isContractor, DEGREE_INJURY_CD))
chisq.test(tab) #sig. contractor death >2 times of non contractor dead
prop.table(tab, margin=1)

#...


###########################################
#find 'contractor' in company
incidents.narrative.contract <- incidents[grep("contract", incidents$IncidentDescription, ignore.case = TRUE), ]
nrow(incidents.narrative.contract) #40

incidents.location.contract <- incidents[grep("contract", incidents$Location, ignore.case = TRUE), ]
nrow(incidents.location.contract) #262

contractors.comp<- incidents %>% mutate(isContractor = case_when((grepl("contract", Location, ignore.case = TRUE) | grepl("contract", IncidentDescription, ignore.case=TRUE))~1, TRUE~0)) 
sum(contractors.comp$isContractor) #297

## how incident profile of contractors is different from the incident profile of non-contractor incidents?

tab <- with(contractors.comp[, c("isContractor", "RiskRating")], table(isContractor, RiskRating))
fisher.test(tab)
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "Hazard")], table(isContractor, Hazard))
tab
chisq.test(tab)
fisher.test(tab)
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "NearMiss")], table(isContractor, NearMiss))
prop.table(tab, margin=1)

tab <- with(contractors.comp[, c("isContractor", "Environmental.Incident")], table(isContractor, Environmental.Incident))
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "Equipment")], table(isContractor, Equipment))
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "Hazard.1")], table(isContractor, Hazard.1))
prop.table(tab, margin=1)

#contractors high
tab <- with(contractors.comp[, c("isContractor", "Injury.Reporting")], table(isContractor, Injury.Reporting))
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "Loss.of.Process")], table(isContractor, Loss.of.Process))
prop.table(tab, margin=1)

tab <- with(contractors.comp[, c("isContractor", "Near.Miss")], table(isContractor, Near.Miss))
prop.table(tab, margin=1)

#contractors low
tab <- with(contractors.comp[, c("isContractor", "Property.Damage")], table(isContractor, Property.Damage))
prop.table(tab, margin=1)

#contractors high
tab <- with(contractors.comp[, c("isContractor", "Security")], table(isContractor, Security))
prop.table(tab, margin=1)

