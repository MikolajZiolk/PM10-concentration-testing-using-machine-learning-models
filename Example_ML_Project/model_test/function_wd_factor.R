
if(!require(tidyverse)){ install.packages("tidyverse") ; library(tidyverse) }


wd_factor <- function(dane, wd, name = "short", ...){
  
  # przedziały kierunku wiatruliczbowe 
  
  rose_breaks <- c(0, 360/32, (1/32 + (1:15 / 16)) * 360, 360)
  
  # etykiety kierunku wiatru krótkie i długie a
  
  if (name == "long") {
    rose_labs <- c(
      "North", "North-Northeast", "Northeast", "East-Northeast",
      "East",  "East-Southeast",  "Southeast", "South-Southeast",
      "South", "South-Southwest", "Southwest", "West-Southwest",
      "West",  "West-Northwest",  "Northwest","North-Northwest",
      "North"
    )
    
    
  } else if (name == "short") {
    
    rose_labs <- c( 
      "N", "NNE", "NE", "ENE",
      "E", "ESE", "SE", "SSE",
      "S", "SSW", "SW", "WSW",
      "W", "WNW", "NW", "NNW",
      "N"
    )
    
  }
  
  # Tworzymy nową zmienną
  
  dane <- dane %>%
    mutate(wd_cut =
             cut({{wd}},
                 breaks = rose_breaks,
                 labels = rose_labs,
                 right = F,
                 include.lowest = T),
           ...	 
    )
  
  # wynik
  
  return(dane) 
}


# dane |> 
#   mutate(wd = wd_factor(wd, name = "short"))


# dane |> wd_factor(wd = wd, name = "short")


# źródło kodu: 
# https://community.rstudio.com/t/convert-wind-direction-degrees-into-factors-in-a-data-frame/14636/2

## TEST ----------------

## Usuń komentarze dla koduc - zaznacz i ctrl+shift+c


## przykłdowe dane

# wind_dir <- data_frame(
#   date = structure(c(13514, 13515, 13516, 13517, 13518, 13519), class = "Date"),
#   SN = c(84084, 84084, 84084, 84084, 84084, 84084),
#   ws = c(21.3, 19.9, 19.3, 21.7, 14.4, 13.9),
#   stopnie = c(34.8, 37.8, 38.2, 15.1, 359, 355)
# )

##  przykład

# wind_dir |>
#   wd_factor(wd = stopnie)
