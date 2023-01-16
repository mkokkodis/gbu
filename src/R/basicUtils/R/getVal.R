#' @export
getVal <- function(curVal){
  if (abs(curVal) > 5){
    return(round(curVal,0))
  }

  if (abs(curVal) > 1){
    return(round(curVal,1))
  }
  if (abs(curVal) < 0.01){
    return(round(curVal,3))
  }
  return(round(curVal,2))
}
