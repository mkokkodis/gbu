#' @export
getColors <- function(colorTheme = 'levander') {
  availCOlors = c('levander', 'blueSky', 'greyGreen', 'green',
                  'purple', 'grey', 'red', 'blueGrey', 'yellow', 'sparklingGreen',
                  'orrange','beige','purple_smooth','greenbeige', 'greenbeige2','unicorn')
  darkColor <- NULL
  ultraLight <- NULL
  extraColor <- NULL
  lightColor <- NULL
  highlightColor <- NULL
  #### darker -> base -> hilght -> light -> ultralight
  #levander: #22223b #4a4e69 #9a8c98 #c9ada7 #f2e9e4
  if (colorTheme == 'levander') {
    darkColor <- '#22223b'
    baseColor <- '#9a8c98'
    highlightColor <- '#c9ada7'
    lightColor <- '#f2e9e4'
  }else if (colorTheme == 'blueSky') { #blueSky: NA #03045e #023e8a  #48cae4 #ade8f4
    baseColor <- '#03045e'
    highlightColor <- '#023e8a'
    lightColor <- '#48cae4'
    ultraLight <- '#ade8f4'
  }else if (colorTheme == 'greyGreen') { #greyGreen: NA #2f3e46  #52796f #84a98c #cad2c5 Extra: #354f52
    baseColor <- '#52796f'
    highlightColor <- '#84a98c'
    lightColor <- '#cad2c5'
    darkColor <- '#2f3e46'
  }else if (colorTheme == 'green') { #green: #081c15 #1b4332 #40916c #74c69d #b7e4c7
    darkColor <- '#081c15'
    baseColor <- '#1b4332'
    highlightColor <- '#40916c'
    lightColor <- '#74c69d'
    ultraLight <- '#b7e4c7'
  }else if (colorTheme == 'purple') { #purples: #10002b #240046 #5a189a #c77dff #e0aaff
    darkColor <- '#10002b'
    baseColor <- '#240046'
    highlightColor <- '#5a189a'
    lightColor <- '#c77dff'
    ultraLight <- '#e0aaff'
  }else if (colorTheme == 'grey') { #greys: NA #212529  #6c757d #ced4da #e9ecef Extra: #343a40
    #darkColor <- '#10002b'
    baseColor <- '#212529'
    highlightColor <- '#6c757d'
    lightColor <- '#ced4da'
    ultraLight <- '#e9ecef'
    extraColor <- '#343a40'
  }else if (colorTheme == 'red') { #reds: #641220 #6e1423 #a11d33 #c71f37 #e01e37
    darkColor <- '#641220'
    baseColor <- '#6e1423'
    highlightColor <- '#a11d33'
    lightColor <- '#c71f37'
    ultraLight <- '#e01e37'
    #extraColor <- '#343a40'
  }else if (colorTheme == 'blueGrey') { #blueGrey: NA #253237  #9db4c0 #c2dfe3 #e0fbfc Extra: #5c6b73
    #darkColor <- '#641220'
    baseColor <- '#253237'
    highlightColor <- '#9db4c0'
    lightColor <- '#c2dfe3'
    ultraLight <- '#e0fbfc'
    extraColor <- '#5c6b73'
  }else if (colorTheme == 'yellow') { # yellow: #76520e  #c9a227 #edc531 #ffe169 Extra: #926c15
    #darkColor <- '#641220'
    baseColor <- '#76520e'
    highlightColor <- '#c9a227'
    lightColor <- '#edc531'
    ultraLight <- '#ffe169'
    extraColor <- '#926c15'
  }else if (colorTheme == 'orrange') { # orrange: #38413f #6f523b #a56336 #b76935  NA Extra: #935e38
    #darkColor <- '#641220'
    baseColor <- '#38413f'
    highlightColor <- '#6f523b'
    lightColor <- '#a56336'
    ultraLight <- '#b76935'
    extraColor <- '#935e38'
  }else if (colorTheme == 'sparklingGreen') { # sparklingGreen: # : #004d00 #00cc00 #99ff99
    baseColor <- '#004d00'
    highlightColor <- '#00cc00'
    lightColor <- '#99ff99'
  }else if (colorTheme == 'purple_smooth') { # palate 4
    darkColor <- '#231942'
    baseColor <- '#5e548e'
    highlightColor <- '#be95c4'
    lightColor <- '#e0b1cb'
  }else if (colorTheme == 'beige') { # palate 3
    darkColor <- '#22223b'
    baseColor <- '#4a4e69'
    highlightColor <- '#c9ada7'
    lightColor <- '#f2e9e4'
  }
  else if (colorTheme == 'greenbeige') { # palate 3
    darkColor <- '#555b6e'
    baseColor <- '#4a4e69'
    highlightColor <- '#ffd6ba'
    lightColor <- '#bee3db'
  }

  else if (colorTheme == 'greenbeige2') { # palate 3
    darkColor <- '#05668d'
    baseColor <- '#028090'
    highlightColor <- '#f0f3bd'
    lightColor <- '#00a896'
    ultraLight = '#02c39a'
  }
  else if (colorTheme == 'unicorn') { # palate 3
    darkColor <- '#64a6bd'
    baseColor <- '#90a8c3'
    highlightColor <- '#d7b9d5'
    lightColor <- '#ada7c9'
    ultraLight = '#f4cae0'
  }
  else {
    return(cat("List of avail color themes:", availCOlors))
  }
  resList <- list("baseColor" = baseColor, "darkColor" = darkColor,
                  "highlightColor" = highlightColor,
                  "lightColor" = lightColor,
                  "ultraLight" = ultraLight, 'extraColor' = extraColor)

  pie(rep(1, length(resList)), col = unlist(resList))
  return(resList)

}


