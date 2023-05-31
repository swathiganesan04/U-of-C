


dbinom(20, size=60, prob=0.33) 
1-pnorm(2)

q10l <- pnorm(q = 6.5, mean = 6, sd = 1.732)
q10u <- pnorm(q = 7.5, mean = 6, sd = 1.732)
q10u - q10l


#To find the mean we take the number of flip x the probability of each 
q12mean <- 25 * 0.5
# The variabnce is calculated using σ2 = Nπ(1-π)
q12var <- (25) * (0.5) * (0.5)
# The standard deviation is the square root of the variance 
q12sd <- sqrt(q12var)
# to find the area under the curve up to 15
q12l <- pnorm(q = 15, mean = q12mean, sd = q12sd)
# to find the area under the curve up to 18 
q12u <- 100 * pnorm(q = 18, mean = q12mean, sd = q12sd)
# Calculate the difference to find the probability and format to only 4 decimal places 
q12bin_form <- format(round((q12u-q12l), 4), nsmall = 4)



n <- 12
sd <- 1.5
var <- sd^2
s_error_delta <- sqrt((var/n)+(var/n))
mean_diff <- 5-4
q5_prob <- pnorm(q = 1.5, mean = mean_diff, sd = s_error_delta)

n_q13 <- 9
sd_q13 <- 0.25
var_q13 <- sd_q13^2
s_error_delta_q13 <- sqrt((var_q13/n_q13)+(var_q13/n_q13))
mean_diff_q13 <- 3-2.8
q13_prob_a <- pnorm(q = 0.5, mean = mean_diff_q13, sd = s_error_delta_q13)
q13_prob_b <- pnorm(q = 0, mean = mean_diff_q13, sd = s_error_delta_q13)
q13_prob_b