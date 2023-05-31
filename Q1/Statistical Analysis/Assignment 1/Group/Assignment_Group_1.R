dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 1/Group/"
test_data <- read.table(paste(dataPath,'test_sample.csv',sep = '/'), header=TRUE)

library(dplyr)

#P(B>A,B>C|B=5)
nrow(filter(test_data, B>A & B>C & B==5)) / nrow(filter(test_data, B==5)) 
#0.7578947

#P(C>A,C>B|C=4)
nrow(filter(test_data, C>A & C>B & C==4)) / nrow(filter(test_data, C==4))
#0.5474138

#P(A>B,A>C)
nrow(filter(test_data, A>B & A>C)) / nrow(test_data) 
#0.293

#P(B>A,B>C)
nrow(filter(test_data, B>A & B>C)) / nrow(test_data) 
#0.36

#P(C>A,C>B)
nrow(filter(test_data, C>A & C>B)) / nrow(test_data) 
#0.347

#Which is the most effective drug among A and B before drug C is available on the market?
prob_a_most_effective = nrow(filter(test_data, A>B)) / nrow(test_data) #0.525
prob_b_most_effective = nrow(filter(test_data, B>A)) / nrow(test_data) #0.475

#Drug A is more effective compared to drug B
  
#Which is the most effective drug among A and B and C?
#Drug B is the best

