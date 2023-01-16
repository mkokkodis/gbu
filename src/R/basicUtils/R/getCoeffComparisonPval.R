#' @export
getCoeffComparisonPval <- function(mean1,mean2,se1,se2,n1,n2){
  var1  <- ((se1 * sqrt(n1))/1.96)^2
  var2  <-   ((se2 * sqrt(n2))/1.96)^2
  
  t = ((mean1 - mean2) - 0)/sqrt(var1/n1 +var2/n2)
  p = 2 * pt(-abs(t), (n1+n2 - 2))
  #cat(t,round(p,4),"\n")
  return(p) #(round(p,5))
  #### Coeffs comparison
  ### https://stats.stackexchange.com/questions/30394/how-to-perform-two-sample-t-tests-in-r-by-inputting-sample-statistics-rather-tha
  #https://www.cyclismo.org/tutorial/R/pValues.html
}