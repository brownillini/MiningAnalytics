#topic modeling using MSHA and Hecla datasets

install.packages("tm")
install.packages("tidyverse")
install.packages("tidytext")
install.packages("topicmodels")
install.packages("LADvis")
install.packages("stopwords")
install.packages("SnowballC")
install.packages("gridExtra")

library(tidyverse)
library(tidytext)
library(topicmodels) #dyn.load("/usr/lib64/atlas/libsatlas.so.3")
library(tm)
library(LDAvis)
library(stopwords)
library(gridExtra)
library(SnowballC)

setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
raw1 <- read.csv("MSHA.injuries.csv", encoding="latin1")
raw2 <-read.csv("incidents.translated.csv", encoding="latin1")

hecla<-read.csv("incidents.translated.preprocessed.csv", encoding="latin1")
nrow(hecla)
sum(hecla$Environmental.Incident)
sum(hecla$Equipment)
sum(hecla$Hazard.1)
sum(hecla$Injury.Reporting)
sum(hecla$Loss.of.Process)
sum(hecla$Property.Damage)
sum(hecla$Security)

#########################################################
#merge description and narrative to get the corpus and dtm for LDA
descriptions <- data.frame(text = c(raw2$IncidentDescription))
descriptions <- data.frame(text = c(raw1$NARRATIVE,raw2$IncidentDescription))
descriptions$id <-seq(1:nrow(descriptions))

token.frequency <- descriptions %>% 
  mutate(text=gsub('[[:punct:]0-9]+', '',gsub('\\\\n|\\.|\\,|\\;',' ',tolower(text)))) %>% 
  select(id, text) %>%
  unnest_tokens(token, text) %>% #token is output column, text is the input 
  mutate(stem = wordStem(token))%>%
  group_by(token) %>%
  mutate(token_freq = n())%>%
  group_by(stem) %>%
  mutate(stem_freq = n())

nrow(token.frequency) #

#stop word removal
head(sort(table(token.frequency$token), decreasing=TRUE), 100)

stopwords <- c(stopwords("en"), c("ee", "employee", "employees", "el", "due", "around"))

sum(token.frequency$token_freq<5) #102455
sum(token.frequency$token %in% stopwords) #2202923

#remove stopwords and low freq words
token.frequency <- token.frequency %>% 
  mutate(stop = case_when(token %in% stopwords~1, TRUE~0)) %>%
  mutate(stop = case_when(stop==1 || nchar(token)<=3 ~1, TRUE~0)) %>%
  filter(token_freq > 5) %>% 
  filter(stop==0) %>%
  select(-stop)

nrow(token.frequency) #5986580

#stemming: not used to keep backing up and back separate
#create Document Term Matrix

#better models without stemming
dtm <- token.frequency %>% 
  cast_dtm(document = id,term = token,value = token_freq)

#worse
dtm <- token.frequency %>% 
  cast_dtm(document = id,term = stem,value = stem_freq)

#filter on tf*idf: https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf

#####################################################################
# 1st try
#create topic model, https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf
# p.13
lda.6 <- LDA(dtm, k = 4) #incident type
#incident type: injury, equipment, property damage, environmental incident

# phi (topic - token distribution matrix) -  topics in rows, tokens in columns:
phi <- posterior(lda.6)$terms %>% as.matrix
phi[,1:8] %>% as_tibble() %>% mutate_if(is.numeric, round, 5) %>% print()

# theta (document - topic distribution matrix) -  documents in rows, topic probs in columns:
theta <- posterior(lda.6)$topics %>% as.matrix
theta[1:8,] %>% as_tibble() %>% mutate_if(is.numeric, round, 5) %>% setNames(paste0('Topic', names(.))) %>% print()

#explore the model
topics <- tidy(lda.6)

# only select top-10 terms per topic based on token probability within a topic
plotinput <- topics %>%
  mutate(topic = as.factor(paste0('Topic',topic))) %>%
  group_by(topic) %>%
  top_n(10, beta) %>% 
  ungroup() %>%
  arrange(topic, -beta)

# plot highest probability terms per topic
names <- levels(unique(plotinput$topic))
colors <- RColorBrewer::brewer.pal(n=length(names),name="Set2")

plist <- list()

for (i in 1:length(names)) {
  d <- subset(plotinput,topic == names[i])[1:10,]
  d$term <- factor(d$term, levels=d[order(d$beta),]$term)
  
  p1 <- ggplot(d, aes(x = term, y = beta, width=0.75)) + 
    labs(y = NULL, x = NULL, fill = NULL) +
    geom_bar(stat = "identity",fill=colors[i]) +
    facet_wrap(~topic) +
    coord_flip() +
    guides(fill=FALSE) +
    theme_bw() + theme(strip.background  = element_blank(),
                       panel.grid.major = element_line(colour = "grey80"),
                       panel.border = element_blank(),
                       axis.ticks = element_line(size = 0),
                       panel.grid.minor.y = element_blank(),
                       panel.grid.major.y = element_blank() ) +
    theme(legend.position="bottom") 
  
  plist[[names[i]]] = p1
}


do.call("grid.arrange", c(plist, ncol=3))

#########################################################
#finding best model https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf
k <- 6 #raw1$NATURE_INJURY
SEED <- 2010
jss_TM <-
  list(VEM = LDA(dtm, k = k, control = list(seed = SEED)),
       VEM_fixed = LDA(dtm, k = k, control = list(estimate.alpha = FALSE, seed = SEED)),
       Gibbs = LDA(dtm, k = k, method = "Gibbs",
                       control = list(seed = SEED, burnin = 1000,
                                        thin = 100, iter = 1000)),
       CTM = CTM(dtm, k = k,
                       control = list(seed = SEED,
                                      var = list(tol = 10^-4), em = list(tol = 10^-3))))
