install.packages("arules")
install.packages("tidyverse")
install.packages("qdapTools")
install.packages("lubridate")

library(arules) #mine rules with one item on the right hand side (rhs).
library(tidyverse) #needed for data conversion to 'transactions'
library(lubridate)
library(qdapTools)


setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
incidents <- read.csv("incidents.translated.preprocessed.csv", na.strings=c("", "NA"))

data <- incidents %>%
  select(Location, Hazard, NearMiss, RiskRating, Status, language, MonthOfOccurance, YearOfOccurance, Environmental.Incident, Equipment, Hazard.1, Injury.Reporting, Loss.of.Process, Near.Miss, Property.Damage, Security)
data <- as.data.frame(unclass(data), stringsAsFactors = TRUE)

sum(complete.cases(data))

data$MonthOfOccurance <- as.factor(data$MonthOfOccurance)
data$YearOfOccurance <- as.factor(data$YearOfOccurance)
data<- data %>% mutate_if(is.integer, as.logical) %>%  mutate_if(is.numeric, as.factor) 

trans <- as(data, "transactions")

#sets <- apriori(trans, parameter = list(support=0.01, target="frequent itemsets"))
#inspect(head(sets, n = 5, by = "support")) 
#inspect(sets)

#adjust support and/or confidence to find useful rules
rules <- apriori(trans, parameter = list(support=0.1, confidence=0.6, target="rules", minlen=2))
#rules.max <- subset(rules, subset=is.maximal(rules))

quality(rules) <- cbind(quality(rules),
                            kulc = interestMeasure(rules, measure = "kulczynski",
                                                   transactions = trans),
                            imbalance = interestMeasure(rules, measure ="imbalance",
                                                        transactions = trans))
#summary(rules)

inspect(head(rules, by="kulc"))

(rules.risk <- subset(rules, subset=rhs %pin% c("RiskRating")))
inspect(head(rules.risk, by="kulc"))

(rules.injury <- subset(rules, subset=rhs %pin% c("Injury")))
inspect(head(rules.injury, by="kulc"))

(rules.eqp <- subset(rules, subset=rhs %pin% c("Equipment")))
(rules.hazard <- subset(rules, subset=rhs %pin% c("Hazard.1")))
(rules.security <- subset(rules, subset=rhs %pin% c("Security")))
(rules.damage <- subset(rules, subset=rhs %pin% c("Damage")))

install.packages("rcompanion")
library(rcompanion)
cramerV(data$NearMiss, data$Injury.Reporting)

data$NearMiss<-ifelse(data$NearMiss=="Yes",1,0)
cor(data$NearMiss, data$Injury.Reporting, use="pairwise.complete.obs")
