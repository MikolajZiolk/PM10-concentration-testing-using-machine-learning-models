## Source code of the CalPM project

# loading all libraries and files

# List of required packages !!!! please add your libraries to the vector !!!!
required_packages <- c("tidyverse", "data.table", "dplyr",
                       "ggpubr", "ranger", "modeldata", "tidymodels",
                       "rpart.plot", "readr","vip", "ggthemes", 
                       "parsnip", "GGally", "skimr", "xgboost",
                       "doParallel", "kernlab")  

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
#print(getwd())

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
  step_rm(date) |> 
  step_dummy(all_nominal_predictors()) |> # wd needs to be numeric 
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors()) # Normalizacja


ops_rec |> prep() |>  bake(train_data) |> glimpse() 



## Linear regression model



## Random Forest model



## Support Vector machine model , SVM

#cel: znalezienie hiperpłaszczyzny maksymalnie separującej dane, co minimalizuje błędy predykcji.
#model SVM
SVM_r_mod <- svm_rbf(
  mode= "regression",
  cost = tune(), # koszt, uniknięcie nadmiernegi dopasowania
  rbf_sigma = tune() # parametr jądra radialnego RBF
) |> 
  set_engine("kernlab")

# Workflow
SVM_wf <- 
  workflow() |> 
  add_model(SVM_r_mod) |> 
  add_recipe(ops_rec)

# Tuning grid, SVM
SVM_grid <- grid_regular(
  cost(), # zakres kosztów np: cost(range = c(-5, 5)),
  rbf_sigma(), # Zakres sigma dla jądra RBF np  rbf_sigma(range = c(-5, 5)),
  levels = 5)

# Metrics
SVM_metrics <- 
  yardstick::metric_set(
    rsq,
    mae,
    rmse)

# 10 Cross validation 
set.seed(123)
SVM_folds <- vfold_cv(
  train_data, 
  v = 10, 
  repeats = 5)

# Parallel computing 
cores = detectCores(logical = FALSE) - 1
cl = makeCluster(cores)
registerDoParallel(cl)

# Hyperparameter tuning
SVM_tune <- tune_grid(
  object = SVM_wf,
  resamples = SVM_folds,
  grid = SVM_grid,
  metrics = SVM_metrics,
  control = control_grid(verbose = TRUE)
)

stopCluster(cl)

#Showing the best candidates for model
top_SVM_models <- 
  SVM_tune |> 
  show_best(metric="rsq", n = Inf) |> 
  arrange(cost) |> 
  mutate(mean = mean |> round(x = _, digits = 3))

top_SVM_models |> gt::gt()

#5 the best models
SVM_tune|> 
  show_best(metric="rsq") |> 
  knitr::kable()

#The best params
SVM_best_params <- SVM_tune |> 
  select_best(metric = "rsq")

# Final Model
SVM_model_final <- SVM_r_mod |>  
  finalize_model(SVM_best_params)

SVM_final_wf <- finalize_workflow(
  SVM_wf,
  SVM_best_params
)

#Fitting with train data
svm_fit <- SVM_final_wf |>
  fit(data = train_data)
#

## XGBoost model ---------------------------------------------------------------



# It is better to load stored data, computation takes too long
load("xgboost_data.RData")

# XGBoost model specification
xgboost_model <- 
  boost_tree(
    mode = "regression",
    trees = 200,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) |> 
  set_engine("xgboost")


# Tuning grid 
xgboost_grid <- 
  grid_regular(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction(),
    levels = 10)


# Workflow
xgboost_wf <- 
  workflow() |> 
  add_model(xgboost_model) |> 
  add_recipe(ops_rec)

# Metrics
xgboost_metrics <- 
  yardstick::metric_set(
    rsq,
    mae,
    rmse)

# Cross validation 
xgboost_folds <- 
  vfold_cv(train_data, v=10, repeats=5)

# Parallel computing 
cores = detectCores(logical = FALSE) - 1
cl = makeCluster(cores)
registerDoParallel(cl)

# Hyperparameter tuning
xgboost_tuned <- 
  tune_grid(
  object = xgboost_wf,
  resamples = xgboost_folds,
  grid = xgboost_grid,
  metrics = xgboost_metrics,
  control = control_grid(verbose = TRUE)
)

stopCluster(cl)

# Selecting best parameters
xgboost_tuned |> 
  show_best(metric = "rsq") |> 
  knitr::kable()

xgboost_best_params <- xgboost_tuned |> 
  select_best(metric = "rsq")


# Final Model
xgboost_model_final <- xgboost_model |>  
  finalize_model(xgboost_best_params)


xgboost_final_wf <- finalize_workflow(
  xgboost_wf,
  xgboost_best_params
)

# Fit
xgboost_final_fit <- fit(xgboost_final_wf, data = train_data)

# Predictions
xgboost_predictions <- predict(xgboost_final_fit, new_data = test_data)

xgboost_results <- test_data |> 
  mutate(
    .pred = xgboost_predictions$.pred) |> 
  select(date, grimm_pm10, .pred)

# Showing metrics of predictions 
xgboost_metrics(xgboost_results, truth = grimm_pm10, estimate = .pred)

# Plot
ggplot(xgboost_results, aes(x = grimm_pm10, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Comparison of Actual vs. Predicted Values",
    x = "Actual values (grimm_pm10)",
    y = "Predictions"
  ) + theme_bw()

# Saving data
save(xgboost_model,
     xgboost_grid,
     xgboost_wf,
     xgboost_metrics,
     xgboost_folds,
     xgboost_tuned,
     xgboost_best_params,
     xgboost_model_final,
     xgboost_final_wf,
     xgboost_final_fit,
     xgboost_predictions,
     xgboost_results,
     file = "xgboost_data.RData")
