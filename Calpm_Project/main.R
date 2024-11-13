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
load("ops.RData") ; ops <-
  ops |>
  na.omit() |> 
  select(-ops_pm10) #external requirement to never use ops_pm10 
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

#Preparing recipe to train models

split <- initial_split(data = ops, prop = 3/4, strata = "grimm_pm10")
train_data <- training(split)
test_data <- testing(split)

# Hellwig method implementation ------------------------------------------------

# Calculating number of all possible combinations of variables (excluding date and second pm10 measure)
(comb_count <- (2^length(setdiff(names(ops %>% select(-c(date))), "grimm_pm10"))) - 1)

# Defining the hellwig() function
hellwig <- function(y, x, method = "pearson") {
  requireNamespace("utils")
  x <- x[sapply(x, is.numeric)]
  x <- as.data.frame(x)
  
  # Calculate correlation matrix
  cm <- stats::cor(x, method = method)
  cd <- stats::cor(x, y, method = method)
  
  k <- sapply(seq(2, ncol(x)), function(i) utils::combn(ncol(x), i, simplify = FALSE))
  k <- do.call("c", k)
  
  hfun <- function(v) {
    sapply(v, function(i) cd[i]^2 / sum(abs(cm[v, i])))
  }
  
  all_combinations <- lapply(k, function(comb) {
    score <- sum(hfun(comb))
    list(combination = paste(names(x)[comb], collapse = "-"), score = score)
  })
  
  all_combinations_df <- data.frame(
    combination = sapply(all_combinations, `[[`, "combination"),
    score = sapply(all_combinations, `[[`, "score"),
    stringsAsFactors = FALSE
  )
  
  return(all_combinations_df)
}

# Defining the target variable and the predictor set
target_var <- c("grimm_pm10")
predictor_set <- setdiff(names(ops %>% select(-c(date))), target_var)
predictor_data <- ops[, predictor_set]
target_data <- ops[[target_var]]

# Calculating estimated information capacity factor for each possible model
all_models <- hellwig(target_data, predictor_data, method = "pearson")

# Selection of variables in three best potential models
best_variables <- unlist(unique(flatten(strsplit(all_models[1:3,]$combination, '-')))) 

# Selection of rejected variables
rejected_predictors <- setdiff(names(ops %>% select(-c(date, grimm_pm10))), best_variables)

# Best model selection
best_model <- all_models |> slice(which.max(score))
best_model

# Initial recipe ---------------------------------------------------------------

ops_rec <- recipe(grimm_pm10  ~., data = train_data) |> 
  step_time(date, features = c("hour")) |>
  step_rm(date) 


ops_rec |> prep() |>  bake(train_data) |> glimpse()



## Linear regression model



## Random Forest model



## Support Vector machine model



## XGBoost model







##metrics












