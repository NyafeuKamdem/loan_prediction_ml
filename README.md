# **Loan Approval Prediction Model**

This project aims to build and evaluate a predictive model that can classify loan applicants as either approved or not approved. Our goal is to provide a tool that helps banks and lending institutions automate and support their decision-making processes.

## **Data Dictionary:**

The dataset I used contains a variety of personal and financial attributes for loan applicants. Here's a quick look at the variables:

| Variable | Definition |
| :---- | :---- |
| Loan\_ID | Unique Loan ID |
| Gender | Male / Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents |
| Education | Applicant Education (Graduate/Undergraduate) |
| Self\_Employed | Self-employed (Y/N) |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Coapplicant income |
| LoanAmount | Loan amount in thousands |
| Loan\_Amount\_Term | Term of loan in months |
| Credit\_History | Credit history meets guidelines |
| Property\_Area | Urban / Semi Urban / Rural |
| Loan\_Status | Loan approved (Y/N) |

## **Our Data Pipeline**

I meticulously prepared the raw data for machine learning through our structured data pipeline:

### **Initial Setup & Exploration:**

* Packages Importation and Data Loading  
  I began by importing standard Python libraries (NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn) essential for data analysis and machine learning. The loans.csv dataset was then loaded, leveraging Google Drive integration.  
* Exploratory Data Analysis (EDA)  
  I thoroughly inspected the dataset's structure, including the number of rows, columns, and data types. I also plotted univariate distributions for key numerical features (applicant\_income, coapplicant\_income, loan\_amount, loan\_amount\_term) and generated a correlation matrix to analyze relationships between these variables.

### **Data Preprocessing for ML:**

* ✅ Step 1: Handling Missing Values  
  I addressed missing values across our dataset. For numerical columns, like LoanAmount, I imputed them using their median values to mitigate outlier impact. For categorical columns (gender, married, self\_employed), I filled missing entries with the mode to preserve their distribution.  
* ✅ Step 2: Feature Engineering  
  I strategically created four new features to enrich our dataset: total\_income (sum of applicant and co-applicant incomes), income\_to\_loan\_ratio (total income divided by loan amount), and logarithmic transformations (loan\_amount\_log, total\_income\_log). These aimed to capture income dynamics and reduce data skewness for our modeling.  
* ✅ Step 3: Outlier Handling  
  I managed outliers in our dataset using a logarithmic method. Any small, artificial outliers that emerged from this transformation were subsequently clipped by applying a defined lower bound, ensuring our data's integrity.  
* ✅ Step 4: Categorical Data Handling  
  In this crucial phase, I converted various descriptive categories (like gender, marital status, and education) into a numerical format. I utilized both One-Hot Encoding (for nominal variables) and Label Encoding (for ordinal variables), which is essential for our machine learning algorithms.  
* ✅ Step 5: String Data Handling  
  Although I included a step for "String Data Handling," I found that there was no specific string data requiring special processing in our pipeline.  
* ✅ Step 6: Dataset Splitting  
  I then divided our prepared dataset into features (X) and the target variable (y). Subsequently, I split it into training and testing sets (typically 80% for training and 20% for testing), ensuring that the distribution of loan approval statuses was maintained consistently across both sets. This provided a robust framework for training and evaluating our predictive models.  
* ✅ Step 7: Feature Scaling  
  For feature scaling, I prepared our numerical features for optimal model performance by addressing data skewness and outliers. I implemented a two-stage scaling approach: RobustScaler was applied to features with skewed distributions and outliers to minimize their impact, followed by MinMaxScaler to bring all features into a consistent scale. This standardization is vital for our machine learning models to learn effectively.

## **Our Model Development and Evaluation**

I developed and rigorously evaluated two primary machine learning models, Logistic Regression and Random Forest, as classifiers for loan approval.

### **1\. Logistic Regression Model Development**

I chose Logistic Regression as our initial predictive model to classify loan applicants.

* **Baseline Evaluation (lr\_model):**  
  * Confusion Matrix:  
    \[\[10 18\]  
    \[ 3 72\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.77 0.36 0.49 28  
    1 0.80 0.96 0.87 75  
    accuracy 0.80 103  
    macro avg 0.78 0.66 0.68 103  
    weighted avg 0.79 0.80 0.77 103  
  * ROC AUC score: 0.698  
* **With SMOTE (sm\_lr\_model):**  
  * Confusion Matrix:  
    \[\[13 15\]  
    \[13 62\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.50 0.46 0.48 28  
    1 0.81 0.83 0.82 75  
    accuracy 0.73 103  
    macro avg 0.65 0.65 0.65 103  
    weighted avg 0.72 0.73 0.72 103  
  * ROC AUC score: 0.698  
* **Hyperparameter Tuned (tuned\_lr\_model):**  
  * Confusion Matrix:  
    \[\[10 18\]  
    \[ 3 72\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.77 0.36 0.49 28  
    1 0.80 0.96 0.87 75  
    accuracy 0.80 103  
    macro avg 0.78 0.66 0.68 103  
    weighted avg 0.79 0.80 0.77 103  
  * ROC AUC score: 0.682  
* **Tuned with SMOTE (sm\_tuned\_lr\_model):**  
  * Confusion Matrix:  
    \[\[15 13\]  
    \[14 61\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.52 0.54 0.53 28  
    1 0.82 0.81 0.82 75  
    accuracy 0.74 103  
    macro avg 0.67 0.67 0.67 103  
    weighted avg 0.74 0.74 0.74 103  
  * ROC AUC score: 0.702

#### **2\. Random Forest Model Development**

Following Logistic Regression, I developed a Random Forest Classifier as an alternative for loan classification.

* **Baseline Evaluation (rf\_model):**  
  * Confusion Matrix:  
    \[\[12 16\]  
    \[ 5 70\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.71 0.43 0.53 28  
    1 0.81 0.93 0.87 75  
    accuracy 0.80 103  
    macro avg 0.76 0.68 0.70 103  
    weighted avg 0.78 0.80 0.78 103  
  * ROC AUC score: 0.719  
* **With SMOTE (sm\_rf\_model):**  
  * Confusion Matrix:  
    \[\[14 14\]  
    \[12 63\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.54 0.50 0.52 28  
    1 0.82 0.84 0.83 75  
    accuracy 0.75 103  
    macro avg 0.68 0.67 0.67 103  
    weighted avg 0.74 0.75 0.74 103  
  * ROC AUC score: 0.647  
* **Hyperparameter Tuned (tuned\_rf\_model):**  
  * Confusion Matrix:  
    \[\[11 17\]  
    \[ 2 73\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.85 0.39 0.54 28  
    1 0.81 0.97 0.88 75  
    accuracy 0.82 103  
    macro avg 0.83 0.68 0.71 103  
    weighted avg 0.82 0.82 0.79 103  
  * ROC AUC score: 0.699  
* **Tuned with SMOTE (sm\_tuned\_rf\_model):**  
  * Confusion Matrix:  
    \[\[15 13\]  
    \[13 62\]\]  
  * Classification Report:  
    precision recall f1-score support  
    0 0.54 0.54 0.54 28  
    1 0.83 0.83 0.83 75  
    accuracy 0.75 103  
    macro avg 0.68 0.68 0.68 103  
    weighted avg 0.75 0.75 0.75 103  
  * ROC AUC score: 0.649

### **Our Final Choice of Model**

After comprehensive evaluation, I identified our **Random Forest model (specifically the baseline rf\_model based on the provided statement)** as the better choice. It demonstrated higher accuracy and overall more reliable performance, especially in maintaining a better balance across predicted classes, despite having a slightly lower AUC score in some iterations compared to others. This suggests it offers a more robust solution for our loan approval classification task.