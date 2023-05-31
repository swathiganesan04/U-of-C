
# Function to return the Nambiar
nambiarNumber <- function(input, i) { 

  # If there is no digit to choose
  if (i >= nchar(input)) {
    return ("")
  }


  # Choose the first digit
  firstDigit <- as.numeric(substr(input, i+1, i+1))
  
  # Chosen digit's parity
  digitParity <- firstDigit %% 2
  
  # To store the sum of the consecutive
  # digits starting from the chosen digit
  sumDigits <- 0
  
  # While there are digits to choose
  while (i < nchar(input)){
    
    # Update the sum
    sumDigits <- (sumDigits + as.numeric(substr(input, i+1, i+1)))
    sumParity <- (sumDigits %% 2)

    # If the parity differs
    if (digitParity != sumParity){
      break}
    i <- i + 1
    
  }
  
  # Return the current sum concatenated with the
  # Nambiar number for the rest of the String
  return (paste(c(sumDigits, nambiarNumber(input, i + 1)), collapse = ""))  

}

# Program to convert decimal number into binary number using recursive function

convert_to_binary <- function(n) {
  bin <- ''
  if(n > 1) {
    bin <- convert_to_binary(as.integer(n/2))
  }
  bin <- paste0(bin, n %% 2)
  return(as.numeric(bin))
}

# Driver code
j<-0
result<-list()

while (j<1000){
input <- sample(100: 999,1)[1][1]
nam_num <- nambiarNumber(input, 0)
result <- append(result, (as.numeric(nam_num)+100)/1000
                 )
#convert_to_binary(as.numeric(nam_num))
j<-j+1

}
result_pseudo <- unlist(result, recursive = FALSE)
hist(result_pseudo)


##################################################

linearCongruentialMethod <- function(X, m, a, c,
                                     randomNums,
                                     noOfRandomNums) { 
  
  
  # Initialize the seed state
  randomNums[1] <- X
  # Traverse to generate required
  # numbers of random numbers
  for (i in 2:noOfRandomNums){
    
    # Follow the linear congruential method
    randomNums[i] <- ((randomNums[i - 1] * a) +
                        c) %% m
  }
  return (randomNums*0.1)
}

X <-  5

# Modulus parameter
m <- 7

# Multiplier term
a <- 3

# Increment term
c <- 3

# Number of Random numbers
# to be generated
noOfRandomNums <- 1000

# To store random numbers
randomNums <- list(integer(noOfRandomNums))

# Function Call
result_true <- linearCongruentialMethod(X, m, a, c,
                                   randomNums[[1]],noOfRandomNums)
hist(result_true)


#######################################################3




res<-data.frame(pseudoSample=result_pseudo,trueSample=result_true)
dataPath <- "/Users/swathiganesan/Desktop/Q1/Statistical Analysis/Assignment 3/Group/"

saveRDS(res,paste(dataPath,'result.rds',sep='/'))




