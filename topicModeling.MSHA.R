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

descriptions <- data.frame(text = c(raw1$NARRATIVE))
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

sum(token.frequency$token_freq<5) 
sum(token.frequency$token %in% stopwords) 

#remove stopwords and low freq words
token.frequency <- token.frequency %>% 
  mutate(stop = case_when(token %in% stopwords~1, TRUE~0)) %>%
  mutate(stop = case_when(stop==1 || nchar(token)<=3 ~1, TRUE~0)) %>%
  filter(token_freq > 5) %>% 
  filter(stop==0) %>%
  select(-stop)

nrow(token.frequency) 

#stemming: not used to keep backing up and back separate
#create Document Term Matrix

#better models without stemming
dtm <- token.frequency %>% 
  cast_dtm(document = id,term = token,value = token_freq)

#worse
#dtm <- token.frequency %>% 
#  cast_dtm(document = id,term = stem,value = stem_freq)

#filter on tf*idf: https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf

#####################################################################
# 1st try
#create topic model, https://cran.r-project.org/web/packages/topicmodels/vignettes/topicmodels.pdf
# p.13
lda.39 <- LDA(dtm, k = 39) #39 nature_injury type in MSHA

#######################################
#39 topics from MSHA data using topicmodels. 1da.39.rds output by R code on UA OnDemand and downloaded into local folder
#took some 30 hours to run on 4 cores.
lda.39<- readRDS(file="lda.39.rds") #

# phi (topic - token distribution matrix) -  topics in rows, tokens in columns:
phi <- posterior(lda.39)$terms %>% as.matrix
phi[,1:10] %>% as_tibble() %>% mutate_if(is.numeric, round, 5) %>% print() # show 10 tokens

# theta (document - topic distribution matrix) -  documents in rows, topic probs in columns:
theta <- posterior(lda.39)$topics %>% as.matrix
View(theta)
#how well does this matches the original?
theta[1:10,] %>% as_tibble() %>% mutate_if(is.numeric, round, 5) %>% setNames(paste0('Topic', names(.))) %>% print()

#explore the model
topics <- tidy(lda.39)

# only select top-10 terms per topic based on token probability within a topic
plotinput <- topics %>%
  mutate(topic = as.factor(paste0('Topic',topic))) %>%
  group_by(topic) %>%
  top_n(10, beta) %>% 
  ungroup() %>%
  arrange(topic, -beta)

View(plotinput)
# plot highest probability terms per topic
names <- levels(unique(plotinput$topic))
#colors <- RColorBrewer::brewer.pal(n=length(names),name="Oranges")

plist <- list()

for (i in 1:length(names)) {
  d <- subset(plotinput,topic == names[i])[1:10,]
  d$term <- factor(d$term, levels=d[order(d$beta),]$term)
  
  p1 <- ggplot(d, aes(x = term, y = beta, width=0.75)) + 
    labs(y = NULL, x = NULL, fill = NULL) +
    geom_bar(stat = "identity",fill="orange") +
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

plist

library(gridExtra)
ml <- marrangeGrob(plist, nrow=2, ncol=4)
ggsave("39topics.pdf", ml)
dev.off()

#do.call("grid.arrange", c(plist, ncol=3))


#https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf
sort(table(raw1$NATURE_INJURY))

for(i in unique(raw1$NATURE_INJURY)){
  print(paste("type =", i))
  print(head(raw1[raw1$NATURE_INJURY==i, ]$NARRATIVE))
}

#topic models of 39 topics reflect more on body parts that are injuried. Not align with Nature_Injuries that well.
#need to improve stopword list