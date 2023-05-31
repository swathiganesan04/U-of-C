dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 6/Group/"

test_data <- read.table(paste(dataPath,'test_sample 2.csv',sep = '/'), header=TRUE)
test_data

library(benford.analysis)

bfd.ts <- benford(test_data$x) 
bfd.ts

head(suspectsTable(bfd.ts, by="difference"),100) 
chisq(bfd.ts)[3]
bfd.ts$stats$mantissa.arc.test


res <- list(numbers=list(37, 85, 90, 89, 30, 29),
            Chi_square_p.value = 4.709645e-19 ,
            Mantissa_p.value = 0.4314,
            MAD = 0.001562239,
            MAD.conformity  = 'Acceptable conformity'
)
res
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
