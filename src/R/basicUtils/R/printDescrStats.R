#' @export
printDescrStats <-function(d,ivs,addLineSpace){

  e <- c('DiversityScore','MoneyDv','HireDv')
  for (col in ivs) {
    tmp <- d[,col]
    meanVal <-  mean(tmp, na.rm=T)
    medVal <-  median(tmp,na.rm=T)
    sdVal <-  sd(tmp, na.rm=T)
    minVal <- min(tmp,na.rm=T)
    maxVal <- max(tmp,na.rm=T)
    cat("&", paste("\\",stringr::str_replace_all(ifelse(!(col  %in% e),tolower(col),col),"_",""), sep=""),
        " & ", getVal(meanVal),
        " & ", getVal(medVal),
        " & ", getVal(sdVal),
        " & ", getVal(minVal),
        " & ",getVal(maxVal),
        "\\\\",
        ifelse(addLineSpace,"\\addlinespace",""),
        "\n")
  }
}
