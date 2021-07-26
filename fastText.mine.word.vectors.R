#create word vector using mining data
library(fastText)

setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
list_params = list(command = 'cbow',
                   lr = 0.1,
                   dim = 300,
                   input = file.path(getwd(), "fatality.reports", "combined.reports.txt"), #doc.txt is where the text 
                   output = file.path(getwd(), "fatal.report.10Kw.vec.bin"),
                   verbose = 2,
                   thread = 1)
res = fasttext_interface(list_params,
                         path_output = file.path(getwd(),"fatal.report.10Kw.vec_logs.txt"),
                         MilliSecs = 100)
