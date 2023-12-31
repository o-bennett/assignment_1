cat("\014")
rm(list=ls())

library(conflicted)
library(tidyverse)
library(jsonlite)
library(dplyr)
library(lubridate)
library(glmnet)
library(rpart)
library(rpart.plot)
library(devtools)
conflicts_prefer(dplyr::filter)
library(knitr)

setwd("C:/Users/Ollie/OneDrive - University of Warwick/Desktop/4th Year/Uni Work/Data Science/Assignment 1 EC349/EC349 Assignment")

load("yelp_review_small.Rda")
load("yelp_user_small.Rda")
business_data <- stream_in(file("yelp_academic_dataset_business.json"))


# Preparing Data ----------------------------------------------------------


# Merging datasets
merged_data <- merge(merge(review_data_small, user_data_small, by = "user_id"), business_data, by = "business_id")

# Removing variables that won't be used in prediction
merged_data_useful <- merged_data %>% 
  select(-c(business_id, user_id, review_id, text, name.x, name.y, address, city, postal_code, latitude, longitude))

# Removing variables that contain > 50% NA or empty entries
threshold <- nrow(merged_data) * 0.5

merged_data_useful <- merged_data_useful %>% 
  select_if(~sum(is.na(.)) < threshold)

# Converting to factor types
merged_data_useful$state <- factor(merged_data_useful$state)
merged_data_useful$is_open <- factor(merged_data_useful$is_open)

# Number of years elite, number of friends
count_elite <- function(x) {
  if (x == "") {
    return(0)
  } else {
    return(length(strsplit(x, ",")[[1]]))
  }
}
merged_data_useful$elite <- sapply(merged_data_useful$elite, count_elite)

count_friends <- function(x) {
  if (x == "None") {
    return(0)
  } else {
    return(length(strsplit(x, ",")[[1]]))
  }
}
merged_data_useful$friends <- sapply(merged_data_useful$friends, count_friends)

# Converting date variables
merged_data_useful$yelping_since <- ymd_hms(merged_data_useful$yelping_since)
merged_data_useful$date <- ymd_hms(merged_data_useful$date)

# Calculate the time difference in days
merged_data_useful$account_age <- as.integer(difftime(merged_data_useful$date, merged_data_useful$yelping_since, units = "days"))

# Removing reviews with a negative time between account creation and review
merged_data_useful <- merged_data_useful %>%
  filter(account_age >= 0)

# Creating factor variables for the year, month and day of week that the reviews were made
merged_data_useful$review_year <- year(merged_data_useful$date)
merged_data_useful$review_month <- month(merged_data_useful$date)
merged_data_useful$review_day_of_week <- wday(merged_data_useful$date, label = TRUE, abbr = FALSE) # Returns the full name of the day

merged_data_useful$review_year <- factor(merged_data_useful$review_year)
merged_data_useful$review_month <- factor(merged_data_useful$review_month)
merged_data_useful$review_day_of_week <- factor(merged_data_useful$review_day_of_week)

merged_data_useful$review_stars <- merged_data_useful$stars.x


# Removing last variables that won't be used in prediction
clean_data <- merged_data_useful %>% 
  select(-c(date, yelping_since, stars.x, categories, compliment_funny))

# Removing states with few reviews (<2)
state_counts <- table(merged_data_useful$state)
states_to_remove <- names(state_counts[state_counts <= 2])
clean_data <- clean_data[!clean_data$state %in% states_to_remove, ]
clean_data$state <- factor(clean_data$state)


# Ridge & LASSO -----------------------------------------------------------


# Converting factor variables to numerical for ridge/lasso
numeric_df <- model.matrix(~ . - 1, data = clean_data)

# Split numerical data into test and training for Ridge and LASSO
set.seed(1)
train <- sample(1:nrow(clean_data), nrow(clean_data) - 10000)
ridge_train <- numeric_df[train,]
ridge_test <- numeric_df[-train,]

# Separate predictors and target variable
ridgex_train <- ridge_train[,-74]
ridgey_train <- ridge_train[,74]
ridgex_test <- ridge_test[,-74]
ridgey_test <- ridge_test[,74]

# Standardise only the predictor variables
mean_ridgex_train <- apply(ridgex_train, 2, mean)
sd_ridgex_train <- apply(ridgex_train, 2, sd)
standardised_ridgex_train <- scale(ridgex_train, center = mean_ridgex_train, scale = sd_ridgex_train)
standardised_ridgex_test <- scale(ridgex_test, center = mean_ridgex_train, scale = sd_ridgex_train)

#Ridge with Cross-Validation
cv.out_ridge <- cv.glmnet(as.matrix(ridgex_train), as.matrix(ridgey_train), alpha = 0, nfolds = 3)
lambda_ridge_cv <- cv.out_ridge$lambda.min

ridge.mod<-glmnet(ridgex_train, ridgey_train, alpha = 0, lambda = lambda_ridge_cv, thresh = 1e-12)

ridge.pred <- predict(ridge.mod, s = lambda_ridge_cv, newx = as.matrix(ridgex_test))
ridge_MSE<- mean((ridge.pred - ridgey_test) ^ 2)

# MSE in training set
ridge_train_pred <- predict(ridge.mod, s = lambda_ridge_cv, newx = as.matrix(ridgex_train))
ridge_train_MSE <- mean((ridge_train_pred - ridgey_train) ^ 2)

#LASSO with Cross-Validation
cv.out_LASSO <- cv.glmnet(as.matrix(ridgex_train), as.matrix(ridgey_train), alpha = 1, nfolds = 3)
lambda_LASSO_cv <- cv.out_LASSO$lambda.min

LASSO.mod<-glmnet(ridgex_train, ridgey_train, alpha = 1, lambda = lambda_LASSO_cv, thresh = 1e-12)

LASSO.pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(ridgex_test))
LASSO_MSE<- mean((LASSO.pred - ridgey_test) ^ 2)

# MSE in training set
LASSO_train_pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(ridgex_train))
LASSO_train_MSE <- mean((LASSO_train_pred - ridgey_train) ^ 2)

# Rounded Ridge and LASSO models
rounded_ridge.pred <- round(ridge.pred)
rounded_LASSO.pred <- round(LASSO.pred)

# MSE in test set
ridge_rounded_MSE <- mean((rounded_ridge.pred - ridgey_test) ^ 2)
LASSO_rounded_MSE <- mean((rounded_LASSO.pred - ridgey_test) ^ 2)

# Accuracy for Rounded Ridge and LASSO models in test set
rounded_ridge_accuracy <- mean(rounded_ridge.pred == ridgey_test)
rounded_LASSO_accuracy <- mean(rounded_LASSO.pred == ridgey_test)

# Rounded Ridge and LASSO models for training set
rounded_ridge_train_pred <- round(ridge_train_pred)
rounded_LASSO_train_pred <- round(LASSO_train_pred)

# MSE for Rounded Ridge and LASSO models in training set
rounded_ridge_train_MSE <- mean((rounded_ridge_train_pred - ridgey_train) ^ 2)
rounded_LASSO_train_MSE <- mean((rounded_LASSO_train_pred - ridgey_train) ^ 2)

# Accuracy for Rounded Ridge and LASSO models in training set
rounded_ridge_train_accuracy <- mean(rounded_ridge_train_pred == ridgey_train)
rounded_LASSO_train_accuracy <- mean(rounded_LASSO_train_pred == ridgey_train)


# OLS ---------------------------------------------------------------------


# Split clean_data into test and training for OLS and trees
OLS_train <- clean_data[train,]
OLSx_train <- OLS_train[,-30]
OLSy_train <- OLS_train[,30]

OLS_test <- clean_data[-train,]
OLSx_test <- OLS_test[,-30]
OLSy_test <- OLS_test[,30]

# OLS
lm_review <- lm(review_stars ~ ., data = OLS_train)
lm_review_predict<-predict(lm_review, newdata = OLSx_test)
lm_review_test_MSE<-mean((lm_review_predict-OLSy_test)^2)

lm_review_train_predict <- predict(lm_review, newdata = OLSx_train)
lm_review_train_MSE <- mean((lm_review_train_predict - OLSy_train) ^ 2)



# Trees -------------------------------------------------------------------


# Converting reviews to categorical for randomly generated reviews and categorical tree
categorical_data <- clean_data
categorical_data$review_stars <- factor(categorical_data$review_stars)

# Split categorical data into test and training
categorical_train <- categorical_data[train,]
categoricalx_train <- categorical_train[,-30]
categoricaly_train <- categorical_train[,30]

categorical_test <- categorical_data[-train,]
categoricalx_test <- categorical_test[,-30]
categoricaly_test <- categorical_test[,30]

# Randomly generated
set.seed(2)
random_reviews <- sample(1:5, 10000, replace = TRUE)
random_mse <- mean((random_reviews - OLSy_test) ^ 2)
random_accuracy <- sum(random_reviews == categoricaly_test)/10000

set.seed(3)
random_reviews_train <- sample(1:5, length(categoricaly_train), replace = TRUE)
random_train_mse <- mean((random_reviews_train - OLSy_train) ^ 2)
random_train_accuracy <- sum(random_reviews_train == categoricaly_train) / length(categoricaly_train)

# Regression tree
reg_tree <- rpart(review_stars ~ ., data = OLS_train)
reg_tree_predict <- predict(reg_tree, OLSx_test)
reg_tree_predict_mse <- mean((reg_tree_predict - OLSy_test) ^ 2)

reg_tree_train_predict <- predict(reg_tree, OLSx_train)
reg_tree_train_MSE <- mean((reg_tree_train_predict - OLSy_train) ^ 2)


# Classification tree
class_tree <- rpart(review_stars ~ ., data = categorical_train)
class_tree_predict <- predict(class_tree, categoricalx_test, type = 'class')
class_tree_accuracy <- sum(class_tree_predict == categoricaly_test)/10000

class_tree_train_predict <- predict(class_tree, categoricalx_train, type = 'class')
class_tree_train_accuracy <- sum(class_tree_train_predict == categoricaly_train) / length(categoricaly_train)
