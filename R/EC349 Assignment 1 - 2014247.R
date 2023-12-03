cat("\014")
rm(list=ls())

library(tidyverse)
library(jsonlite)
library(dplyr)
library(lubridate)
library(glmnet)
library(rpart)
library(rpart.plot)

setwd("C:/Users/Ollie/OneDrive - University of Warwick/Desktop/4th Year/Uni Work/Data Science/Assignment 1 EC349/EC349 Assignment")

load("yelp_review_small.Rda")
load("yelp_user_small.Rda")
business_data <- stream_in(file("yelp_academic_dataset_business.json"))

#Merging datasets
merged_data <- merge(merge(review_data_small, user_data_small, by = "user_id"), business_data, by = "business_id")

# Removing variables that won't be used in prediction
merged_data_useful <- merged_data %>% 
  select(-c(business_id, user_id, review_id, text, name.x, name.y, address, city, postal_code, latitude, longitude))

# Removing variables that contain > 50% NA or empty entries
threshold <- nrow(merged_data) * 0.5

merged_data_useful <- merged_data_useful %>% 
  select_if(~sum(is.na(.)) < threshold)

#merged_data_useful <- merged_data_useful %>% 
 # select_if(~!is.numeric(.) | sum(. == 0, na.rm = TRUE) < threshold)

#merged_data_useful <- merged_data_useful %>% 
 # select_if(~!is.character(.) | sum(. == "", na.rm = TRUE) < threshold)

#merged_data_useful <- merged_data_useful %>% 
 # select_if(~!is.character(.) | sum(. == "None", na.rm = TRUE) < threshold)

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
  } else {eu
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

merged_data_useful$review_stars <- merged_data_useful$stars.x


# Removing last variables that won't be used in prediction
clean_data <- merged_data_useful %>% 
  select(-c(date, yelping_since, stars.x, categories))

# Converting factor variables to numerical for ridge/lasso
numeric_df <- model.matrix(~ . - 1, data = clean_data)


# Split numerical data into test and training for Ridge and LASSO
set.seed(1)
train <- sample(1:nrow(clean_data), nrow(clean_data)-10000)
ridge_train <- numeric_df[train,]
ridgex_train <- ridge_train[,-82]
ridgey_train <- ridge_train[,82]

ridge_test <- numeric_df[-train,]
ridgex_test <- ridge_test[,-82]
ridgey_test <- ridge_test[,82]

#Ridge with Cross-Validation
cv.out <- cv.glmnet(as.matrix(ridgex_train), as.matrix(ridgey_train), alpha = 0, nfolds = 3)
plot(cv.out)
lambda_ridge_cv <- cv.out$lambda.min

ridge.mod<-glmnet(ridgex_train, ridgey_train, alpha = 0, lambda = lambda_ridge_cv, thresh = 1e-12)

ridge.pred <- predict(ridge.mod, s = lambda_ridge_cv, newx = as.matrix(ridgex_test))
ridge_MSE<- mean((ridge.pred - ridgey_test) ^ 2)

#LASSO with Cross-Validation
cv.out <- cv.glmnet(as.matrix(ridgex_train), as.matrix(ridgey_train), alpha = 1, nfolds = 3)
plot(cv.out)
lambda_LASSO_cv <- cv.out$lambda.min

LASSO.mod<-glmnet(ridgex_train, ridgey_train, alpha = 1, lambda = lambda_LASSO_cv, thresh = 1e-12)

LASSO.pred <- predict(LASSO.mod, s = lambda_LASSO_cv, newx = as.matrix(ridgex_test))
LASSO_MSE<- mean((LASSO.pred - ridgey_test) ^ 2)

# Split clean_data into test and training for OLS and trees
OLS_train <- clean_data[train,]
OLSx_train <- OLS_train[,-31]
OLSy_train <- OLS_train[,31]

OLS_test <- clean_data[-train,]
OLSx_test <- OLS_test[,-31]
OLSy_test <- OLS_test[,31]

# OLS
lm_review <- lm(review_stars ~ ., data = OLS_train)
lm_review_predict<-predict(lm_review, newdata = OLSx_test)
lm_review_test_MSE<-mean((lm_review_predict-OLSy_test)^2)

# Converting reviews to categorical for randomly generated reviews and categorical tree
categorical_data <- clean_data
categorical_data$review_stars <- factor(categorical_data$review_stars)

# Split categorical data into test and training
categorical_train <- categorical_data[train,]
categoricalx_train <- categorical_train[,-31]
categoricaly_train <- categorical_train[,31]

categorical_test <- categorical_data[-train,]
categoricalx_test <- categorical_test[,-31]
categoricaly_test <- categorical_test[,31]

# Randomly generated
set.seed(2)
random_reviews <- sample(1:5, 10000, replace = TRUE)
random_mse <- mean((random_reviews - OLSy_test) ^ 2)
random_accuracy <- sum(random_reviews == categoricaly_test)/10000

# Regression tree
reg_tree <- rpart(review_stars ~ ., data = OLS_train)
reg_tree_predict <- predict(reg_tree, OLSx_test)
reg_tree_predict_mse <- mean((reg_tree_predict - OLSy_test) ^ 2)


# Classification tree
class_tree <- rpart(review_stars ~ ., data = categorical_train)
class_tree_predict <- predict(class_tree, categoricalx_test, type = 'class')
class_tree_accuracy <- sum(class_tree_predict == categoricaly_test)/10000




#KEEP VARIABLES WITH MANY 0 ENTRIES - could be a strong indicator
#scrap variables with many NA or emptry entries


#city variable is no good as quite a cities only have one review
# Keep the state variable (factor)

# have reactions as a fraction of total reactions - bad reviews with lots of reactions could have more good reactions than good reviews with less reactions just because they have been viewed more
# add a new variable for total number of reactions




# clean data
# split into training and test
# do ols to compare it - SKIP OLS - CAN JUSTIFY NOT INCLUDING IT (WE CARE ABOUT PRECITION NOT BIAS/INTERPRETABILITY)
# ridge/lasso estimator + cross validation to find lambda?
# use prediction trees
# pick model with lowest mse
# use model to predict test