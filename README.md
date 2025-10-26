# LendingClub Loan Approval Analysis and Modeling

## Project Overview
This project explores two different machine learning paradigms for loan approval decision-making using the LendingClub dataset:

1. **Deep Learning Model (Supervised Learning):**  
   Predicts whether a loan will be fully paid or defaulted.

2. **Offline Reinforcement Learning Agent (Policy Learning):**  
   Learns an optimal decision policy (approve/deny) from historical data, maximizing long-term financial reward.

---

## Dataset
- File used: `accepted_2007_to_2018Q4.csv`  
- Sample for initial EDA: ~2% of dataset for faster exploration  
- Preprocessing steps include:
  - Dropping columns with >40% missing values
  - Removing data leakage columns (like `total_pymnt`)
  - Encoding categorical variables using one-hot encoding
  - Scaling numeric features with `StandardScaler`

---

## Task 1: Exploratory Data Analysis (EDA)

**EDA Highlights:**

- **Missing Values:** Identified top 20 columns with most missing data.  
- **Loan Status Distribution:** Most loans are fully paid, a smaller fraction defaulted.  
- **Interest Rate Distribution:** Majority of loans fall in a 5–25% interest range.  
- **Loan Amount Distribution:** Most loans are clustered between $5k–$20k.  
- **Annual Income Distribution:** Right-skewed distribution; log-transformed for visualization.  
- **Correlation Heatmap:** `loan_amnt`, `int_rate`, `annual_inc` show moderate correlations with each other.

*Inference:* The dataset is highly imbalanced in terms of default vs fully paid loans; numeric features show variability and some predictive power.

---

## Task 2: Feature Engineering & Selection

- Dropped columns with >40% missing values.  
- Selected meaningful features such as `loan_amnt`, `term`, `int_rate`, `grade`, `emp_length`, `home_ownership`, `annual_inc`, `dti`, `delinq_2yrs`, `revol_bal`, etc.  
- Cleaned numeric features:
  - Converted `int_rate` and `revol_util` to floats
  - Extracted numeric values from `term` and `emp_length`  
- Encoded categorical variables using one-hot encoding.  
- Mapped `loan_status` to binary target `loan_status_binary` (0 = non-default, 1 = default).  

*Inference:* Dataset is now clean, numeric, and ready for modeling with no missing values in critical fields.

---

## Task 3: Data Cleaning & Preprocessing

- Split dataset into features (`X`) and target (`y`).  
- Handled remaining missing numeric values using median imputation.  
- Performed train-test split (80-20) with stratification to maintain class balance.  
- Applied `StandardScaler` to scale numeric features.  

*Result:* Feature matrices and target vectors are ready for both deep learning and reinforcement learning tasks.

---

## Task 4: Modeling

### 1. Deep Learning Model (MLP)

- Metrics on test set:
  - **AUC:** 0.714  
  - **F1-score:** 0.666  
- Classification report shows balanced precision and recall for both classes.  

*Interpretation:*  
The DL model discriminates reasonably well between defaulted and fully paid loans. AUC measures separability, while F1-score balances false positives and false negatives.

---

### 2. Offline RL Agent (DQN)

- **Estimated Policy Value:** 96.26  
- Trained using historical data, with the environment designed to reward approvals of good loans and penalize losses from defaults.  

*Interpretation:*  
The RL agent optimizes for long-term financial reward rather than classification accuracy. The policy learned balances risk and potential profit effectively.

---

### 3. Comparison of Policies

- **DL Model Policy:** Approve if predicted default probability < 0.5 (conservative approach).  
- **RL Agent Policy:** Approves loans based on expected reward considering interest vs. default risk (strategic profit-aware approach).  
- Example differences: High-risk applicants may be rejected by DL but approved by RL if potential profit is high.

---

## Future Steps

1. Deploy RL agent for pilot testing on new applications with human oversight.  
2. Collect additional financial features (credit utilization, loan history, income details).  
3. Explore advanced RL algorithms: BCQ, TD3+BC, Decision Transformer.  
4. Implement cost-sensitive DL models to align predictions with financial outcomes.  
5. Deploy hybrid system: DL for risk estimation + RL for decision optimization.  

*Limitations:*  
- Both models depend on historical data; potential biases may persist.  
- RL requires well-designed rewards to avoid unintended behavior.  
- Offline RL assumes historical data adequately represents all scenarios.

---

## Final Thoughts

- **Deep Learning Model:** Risk-aware, focuses on minimizing defaults.  
- **Offline RL Agent:** Profit-aware, focuses on maximizing long-term financial return.  
- **Hybrid approach:** Combining both could lead to intelligent, fair, and financially optimized lending decisions.

---

## Files in the Repository

- `accepted_2007_to_2018Q4.csv` – Raw LendingClub data  
- `mlp_lendingclub_model.h5` – Trained deep learning model  
- `mlp_lendingclub_balanced.h5` – Trained balanced MLP  
- `scaler.joblib` / `scaler_balanced.joblib` – Feature scalers  
- `dqn_lendingclub_agent.zip` – Trained RL agent  
- `model_analysis.ipynb` - Jupyter Notebook (contains whole code)
---
