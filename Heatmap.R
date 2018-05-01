M <- matrix(
  c("","",""),
  nrow=1,
  ncol=3)
colnames(M) <- c("nVariables","nTrees","MSE")
for(x in 1:10){
  X <- x
  for(y in 1:10){
    Y <- y
    z <- (1/(3*X+2*Y))
    M <- rbind(M, c(X,Y,z))
  }
}
M <- M[-1,]

