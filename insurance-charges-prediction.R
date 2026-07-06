---
title: "Predicting Individual Medical Insurance Charges"
subtitle: "A six-model regression comparison in R on demographic and health-behavior features"
author: "eli mend"
output: html_document
---

# Overview

This project predicts individual medical insurance charges from a set of demographic and health-behavior features — age, sex, BMI, number of children, smoking status, and region — using six regression models. The goal is to compare model families on the same underlying dataset and identify which combination of algorithm and feature representation minimizes prediction error while remaining interpretable enough for stakeholder communication.

**Models compared:** Multiple Linear Regression, Generalized Linear Model, Decision Tree, Support Vector Machine, Random Forest, and Gradient Boosting Machine.

**Evaluation:** RMSE and R² on a held-out test set, plus 10-fold cross-validated RMSE.

**Dataset:** `insurance.csv` (1,338 observations, 7 variables) — publicly available on Kaggle and mirrored in R's `MachineLearning` teaching materials.

---

```{r libraries}
# Read in libraries needed:
library(caret)
library(randomForest)
library(rpart)
library(ggplot2)
library(e1071)
library(gbm)
library(dplyr)
library(GGally)
```

```{r EDA}
# Read in the file 
insurance <- read.csv("insurance.csv")

# Performing EDA:
# Display the structure of the dataset
str(insurance)

# Summarize the variables in the dataset
summary(insurance)

# View the summary statistics of the continuous variables
summary(insurance[c("age", "bmi", "children", "charges")])

# Plot a histogram of the charges variable
ggplot(insurance, aes(x = charges)) + 
  geom_histogram(fill = "blue", color = "black", binwidth = 1000) + 
  labs(x = "Charges", y = "Frequency", title = "Distribution of Charges")

# The histogram shows most insurance charges are under 200000
# Skewed


# Plot a scatter plot of age vs. charges
ggplot(insurance, aes(x = age, y = charges)) + 
  geom_point(color = "blue") + 
  labs(x = "Age", y = "Charges", title = "Age vs. Charges")

# The scatter plot shows the older the age the more the charges start off


# Create a boxplot of charges by region
ggplot(insurance, aes(x = region, y = charges, fill = region)) + 
  geom_boxplot() +
  labs(x = "Region", y = "Charges") +
  theme_minimal()

# The boxplot shows the different regions with Southeast having the most charges


# Plot a density plot of bmi by smoker
ggplot(insurance, aes(x = bmi, fill = smoker)) + 
  geom_density(alpha = 0.5) + 
  labs(x = "BMI", y = "Density", title = "BMI by Smoker") + 
  scale_fill_manual(values = c("grey", "orange"), name = "Smoker")

# Slightly more non-smokers than smokers


# Create a violin plot of charges by smoker status
ggplot(insurance, aes(x = smoker, y = charges, fill = smoker)) + 
  geom_violin() +
  labs(x = "Smoker Status", y = "Charges") +
  theme_minimal()

# Smokers experience more charges

# Create a boxplot of BMI by region 
ggplot(insurance, aes(x = region, y = bmi, color = region)) + 
  geom_boxplot() +
  labs(x = "Region", y = "BMI") +
  theme_minimal()

# Exploring region and bmi: Southeast has the highest BMI

# Create a stacked bar for regions and how many children there are in each
ggplot(insurance, aes(x=region, fill=factor(children))) +
  geom_bar() +
  ggtitle("Region and Children") +
  xlab("Region") +
  ylab("Count") +
  scale_fill_manual(values=c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))

# Shows that the Southeast region has the most children in general



# Show a table in how medians compare in each region, can get a better look of values:
insurance %>%
  group_by(region) %>%
  summarise(median_age = median(age),
            median_bmi = median(bmi),
            median_children = median(children),
            median_charges = median(charges))

# Show a table of how means compare in each region:
insurance %>%
  group_by(region) %>%
  summarise(mean_age = mean(age),
            mean_bmi = mean(bmi),
            mean_children = mean(children),
            mean_charges = mean(charges))

# Put these in a Excel file to make the table look better in paper



# Change catergorical data to numeric with encodings
insurance_encoded <- insurance %>%
  mutate(sex = recode(sex, "male" = "0", "female" = "1"), 
         smoker = recode(smoker, "no" = "0", "yes" = "1"),
         region = recode(region, "northeast" = "1", "northwest" = "2",
                         "southeast" = "3", "southwest" = "4"))
        


# Perform correlation:
# Doing correlation on numeric only variables first
insurance_numeric <- insurance[,c(1,3:4,7)]
cor(insurance_numeric)

# According to the correlation above, none are correlated to each other (of the numeric variables)

# Scatterplot matrix of numeric variables:
ggpairs(insurance_numeric)
# We can see from the correlation matrix, that age and bmi are highly correlated
# to charges while children have a smaller correlation. 
# We can also see how these numeric variables range - the charges is heavily skewed right
# While age and bmi are not skewed.
# We see that the distrution of children is generally the within mimumum - Q3

# Scatterplot matrix of the entire dataset:
ggpairs(insurance)
```

## Models

Models include multiple linear regression, generalized linear, decision tree, 
gradient boost, regression support vector, and random forest.

```{r training and test sets}
# Convert categorical variables to factors
insurance$sex <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)

# Split the data into training and testing sets
trainIndex <- createDataPartition(insurance$charges, p = 0.7, list = FALSE)
trainData <- insurance[trainIndex, ]
testData <- insurance[-trainIndex, ]
```

```{r evaluation metrics}
# Create a function to compute RMSE and R-squared
eval_metrics <- function(actual, predicted){
  RMSE <- sqrt(mean((actual - predicted)^2))
  R_squared <- cor(actual, predicted)^2
  return(list(RMSE = RMSE, R_squared = R_squared))
}
```

```{r multiple regression}
# Set seed for reproducibility:
set.seed(123)

# Multiple linear regression model
lm_model <- lm(charges ~ ., data = trainData)

# Prediction against the test set
lm_preds <- predict(lm_model, newdata = testData)

# Compute RMSE and R-squared
lm_metrics <- eval_metrics(testData$charges, lm_preds)

# Compute CV_RMSE
lm_cv <- sqrt(mean((trainData$charges - predict(lm_model, trainData))^2))
```

```{r generalized linear}
# Set seed for reproducibility:
set.seed(123)

# Generalized linear model
glm_model <- glm(charges ~ ., data = trainData, family = "gaussian")

# Prediction against the test set
glm_preds <- predict(glm_model, newdata = testData)

# Evaluate model performance
glm_metrics <- eval_metrics(testData$charges, glm_preds)
glm_cv <- sqrt(mean((trainData$charges - predict(glm_model, trainData))^2))
```

```{r decision tree}
# Decision tree model
tree_model <- rpart(charges ~ ., data = trainData)

# Prediction against the test set
tree_preds <- predict(tree_model, newdata = testData)

# Evaluate model performance
tree_metrics <- eval_metrics(testData$charges, tree_preds)
tree_cv <- sqrt(mean((trainData$charges - predict(tree_model, trainData))^2))
```

```{r support vector model}
# Create and train SVR model
svm_model <- svm(charges ~ ., data = trainData, kernel = "radial")

# Make predictions on the test data
predicted_charges <- predict(svm_model, testData)

# Evaluate model performance
svm_metrics <- eval_metrics(testData$charges, predicted_charges)
svm_cv <- sqrt(mean((trainData$charges - predict(svm_model, trainData))^2))
```


```{r dummy variables}
# Convert factor variables into dummy variables
insurance$sex <- as.factor(insurance$sex)
insurance$smoker <- as.factor(insurance$smoker)
insurance$region <- as.factor(insurance$region)
dummy_model <- dummyVars(charges ~ ., data = insurance)
insurance_dummy <- predict(dummy_model, newdata = insurance)

insurance_dummy <- as.data.frame(insurance_dummy)

# Add the charges column from the original data
insurance_dummy$charges <- insurance$charges
```

```{r training and test set dummy}
# Split the data into training and testing sets
set.seed(123)
trainIndex1 <- createDataPartition(insurance_dummy$charges, p = 0.7, list = FALSE)
trainData1 <- insurance_dummy[trainIndex1, ]
testData1 <- insurance_dummy[-trainIndex1, ]
```

```{r random forest}
# Random forest model
rf_model <- randomForest(charges ~ ., data = trainData1, ntree = 1000)

# Prediction against the test set
rf_preds <- predict(rf_model, newdata = testData1)

# Evaluate model performance
rf_metrics <- eval_metrics(testData1$charges, rf_preds)
rf_cv <- sqrt(mean((trainData1$charges - predict(rf_model, trainData1))^2))
```

```{r gradient boosted}
# Gradient boosted model
gbm_model <- gbm(charges ~ ., data = trainData1, n.trees = 1000, interaction.depth = 4, shrinkage = 0.01)

# Prediction against the test set
gbm_preds <- predict.gbm(gbm_model, newdata = testData1, n.trees = 1000)

# Evaluate model performance
gbm_metrics <- eval_metrics(testData1$charges, gbm_preds)
gbm_cv <- sqrt(mean((trainData1$charges - predict(gbm_model, trainData1))^2))
```

## Compare models

```{r comparing models}
# display the results in a table
results_table <- data.frame(
  Model = c("Multiple Linear Regression", "Generalized Linear Model", "Decision Tree", "Gradient Boost","Support Vector", "Random Forest"),
  RMSE = c(lm_metrics$RMSE, glm_metrics$RMSE, tree_metrics$RMSE, gbm_metrics$RMSE, svm_metrics$RMSE, rf_metrics$RMSE),
  CV_RMSE = c(lm_cv, glm_cv, tree_cv, gbm_cv, svm_cv, rf_cv),
  R_squared = c(lm_metrics$R_squared, glm_metrics$R_squared, tree_metrics$R_squared, gbm_metrics$R_squared, svm_metrics$R_squared, rf_metrics$R_squared)
)
results_table
```
## Visualizing the best model

```{r random forest visualization}
# Plot Variable Importance for Random Forest model
varImpPlot(rf_model, main = "Variable Importance - Random Forest")
```