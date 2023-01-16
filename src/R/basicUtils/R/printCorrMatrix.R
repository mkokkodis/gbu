#' @export
printCorrMatrix <- function(df,vars){
  tmp <- df[,vars]
  newCols <- c()
  for (col in c(ivs,focalVar)){
    newCols <- c(newCols,paste('\\',tolower(col),sep=""))
  }
  colnames(tmp) <- newCols
  mcor<-round(cor(tmp),2)
  upper<-mcor
  upper[upper.tri(mcor)]<-""
  upper<-as.data.frame(upper)
  newCols <- c()
  for (col in colnames(tmp)){
    newCols <- c(newCols,paste('\\multicolumn{1}{p{0.6in}}{',col,"}",sep=""))
  }
  colnames(upper) <- newCols
  
  print(xtable(upper),sanitize.text.function=function(x){x})
}