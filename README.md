# Predicting Individual Medical Insurance Charges
 
A six-model regression comparison in R, predicting individual insurance charges from demographic and health-behavior features.
 
## Problem
 
Given demographic and health-behavior features about an individual (age, sex, BMI, number of dependents, smoking status, US region), predict their annual medical insurance charges. The predictions can inform actuarial pricing decisions, help individuals estimate expected costs, and identify which features drive the largest premium differences.
 
## Dataset
 
`insurance.csv` — 1,338 observations, 7 variables. Publicly available on Kaggle.
 
| Feature   | Type       | Description                                |
|-----------|------------|--------------------------------------------|
| age       | numeric    | Primary beneficiary age                    |
| sex       | categorical| Insurance contractor gender                |
| bmi       | numeric    | Body mass index                            |
| children  | numeric    | Number of dependents covered               |
| smoker    | categorical| Smoker (yes / no)                          |
| region    | categorical| US census region (NE / NW / SE / SW)       |
| charges   | numeric    | **Target** — individual medical costs (USD) |
 
## Approach
 
1. **Exploratory data analysis** with `ggplot2` — distribution of charges (right-skewed), scatter of age vs. charges (fanning pattern), boxplots by region and smoking status. Smoking status emerges as the single strongest predictor.
2. **Train/test split** — 80/20 with `set.seed(123)` for reproducibility.
3. **Evaluation function** — custom `eval_metrics()` returning RMSE and R² for each model on the test set, plus training-set RMSE for a rough overfitting check.
4. **Six regression models trained on identical splits:**
   - Multiple Linear Regression
   - Generalized Linear Model
   - Decision Tree (`rpart`)
   - Gradient Boosting Machine (`gbm`)
   - Support Vector Machine (`e1071`)
   - Random Forest (`randomForest`) — with additional one-hot encoding of categorical features
5. **Cross-validation** — 10-fold CV RMSE for each model as a robustness check against a single train/test split.
6. **Variable importance** — extracted from the Random Forest fit and visualized.
## Results
 
| Model                       | Test RMSE | CV RMSE  | R²     |
|-----------------------------|-----------|----------|--------|
| Multiple Linear Regression  | 5,930     | 6,095    | 0.762  |
| Generalized Linear Model    | 5,930     | 6,095    | 0.762  |
| Decision Tree               | 5,068     | 4,848    | 0.826  |
| Gradient Boost              | 4,787     | 3,816    | 0.837  |
| **Support Vector**          | **4,763** | 4,822    | **0.850** |
| Random Forest               | 4,987     | **3,347**| 0.826  |
 
**Takeaways:**
 
- **SVM wins on test RMSE and R²**, but the difference between SVM (4,763) and Gradient Boost (4,787) is small enough to be within noise on a single split.
- **Random Forest wins on cross-validated RMSE** (3,347), suggesting the single test-split result under-represents its true performance. Where the two disagree, CV_RMSE is usually the more reliable estimate.
- **Multiple Linear and GLM are identical** — with a Gaussian family and identity link, `glm()` degenerates to `lm()`. The identical numbers are a correctness check, not a coincidence.
- **Tree-based models beat linear models by ~15–20% on RMSE**, consistent with the presence of strong interactions in the data (particularly smoking × BMI × age).
## Tech Stack
 
- R
- `caret` — evaluation and cross-validation
- `randomForest`, `rpart`, `gbm`, `e1071` — model implementations
- `ggplot2`, `GGally` — visualization
- `dplyr` — data manipulation
## Files
 
- `insurance-charges-prediction.Rmd` — full analysis notebook (Quarto/RMarkdown)
- `insurance-charges-prediction.html` — knit output with rendered plots and tables
- `insurance.csv` — dataset
## How to Run
 
```r
install.packages(c("caret", "randomForest", "rpart", "ggplot2", "e1071", "gbm", "dplyr", "GGally"))
# In RStudio, open the .Rmd file and Knit to HTML
```
 
## Context
 
This was my undergraduate research capstone. Since then, my professional work has centered on production ETL pipelines rather than model comparison, but the modeling instincts this project built — evaluating multiple candidates on the same split, treating CV as more reliable than any single test result, extracting variable importance for interpretability — carried directly forward.
