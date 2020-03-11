#!/Library/Frameworks/R.framework/Resources/Rscript
setwd("/Users/shua784/Dropbox/PNNL/Project/HFR-thermal/outputs/TH-100x100x2-thermal/temp_100H/")
library(gtools)
file.rename(list.files(pattern = "*.png"), paste0("sorted/100H.", sprintf("%04d", 0:365), ".png"))
