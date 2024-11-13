## Source code of the CalPM project

# loading all libraries and files

# List of required packages !!!! please add your libraries to the vector !!!!
required_packages <- c("tidyverse", "data.table", "dplyr",
                       "ggpubr", "ranger", "modeldata", "tidymodels",
                       "rpart.plot", "readr","vip", "ggthemes", 
                       "parsnip", "GGally", "skimr")  

# Function to run packages and install missing ones
install_if_missing <- function(packages) {
  missing_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
  if(length(missing_packages)) {
    install.packages(missing_packages)
  }
  invisible(lapply(packages, library, character.only = TRUE))
}

install_if_missing(required_packages)
tidymodels_prefer()
set.seed(123)
#set working directiory to the one where document exists
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#don't modify the ops file as this is our input we should always derive from
load("ops.RData") ; ops <- ops |> na.omit()
#ops |> glimpse()

##data preparation
#transforming wind to quality variable
wind_set_dir <- function(kat) {
  directions <- c("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
  index <- floor((kat + 11.25) / 22.5) %% 16 + 1
  
  directions[index]
}

ops <- ops |> 
  mutate(wd = wind_set_dir(wd))

##Hellwig method implementation


#Preparing recipe to train models

split <- initial_split(data = ops, prop = 3/4, strata = "grimm_pm10")
train_data <- training(split)
test_data <- testing(split)



ops_rec <- recipe(grimm_pm10  ~., data = train_data) |> 
  update_role(ops_pm10, new_role = "ID") |> 
  step_time(date, features = c("hour")) |>
  step_rm(date) 


ops_rec |> prep() |>  bake(train_data) |> glimpse()



## Linear regression model



## Random Forest model



## Support Vector machine model , SVM
#cel: znalezienie hiperpłaszczyzny maksymalnie separującej dane, co minimalizuje błędy predykcji.
#model SVM
svm_spec <- svm_rbf(
  mode= "regression",
  cost = tune(), # koszt
  rbf_sigma = tune() # parametr jądra RBF
) |> 
  set_engine("kernlab")



## XGBoost model







##metrics












