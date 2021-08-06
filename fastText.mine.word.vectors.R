#create word vector using mining data
library(fastText)

setwd("C:/Users/hongcui/Documents/research/2021ALPHA with Brown/R")
list_params = list(command = 'cbow', #skip gram is ok too
                   lr = 0.1,
                   dim = 300,
                   input = file.path(getwd(), "fatality.reports", "combined.reports.txt"), #need to direct to the folder where combined.reports.txt is
                   output = file.path(getwd(), "fatal.report.10Kw.vec.bin"),
                   verbose = 2,
                   thread = 1)
res = fasttext_interface(list_params,
                         path_output = file.path(getwd(),"fatal.report.10Kw.vec_logs.txt"),
                         MilliSecs = 100)
