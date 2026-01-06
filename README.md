# AI Enabled Visa Status Prediction and Processing time Estimator
**Milestone 1:** Data Collection & Preprocessing 

  **Objective:** Build structured, cleaned dataset for modeling.
  
  **Tasks Performed:**
  1. **Data collection**: Searched multiple reliable websites and downloaded historical records related to visa processing and merged those sources into a single, consistent             dataset (aligned columns, removed duplicates). The target label (Processing_Days) was generated during the creation of dataset.
  2. **Handled missing values**: Identified which columns had missing data and how much was missing and applied appropiate strategies. For columns with numerical datatype, the           missing rows were filled with median and for columns with string datatype, the missing rows were filled with "Unkown" or mode value.
  3. **Encoded categorical features**: The text categorical data was converted into numeric form using one hot encoding, so that models can use them properly and also ensured that       encodings didn’t introduce unintended ordinal relationships where none exist.

**Milestone 2**: Exploratory Data Analysis & Feature Engineering

  **Objective**: Derive insights and prepare features.

  **Tasks Performed**:

   1. **Exploratory Data Analysis (EDA)**: Computed descriptive statistics for Processing_Days to understand central tendency, dispersion, and skewness. Visualized the distribution using histograms and KDE plots, identified extreme values through boxplots, and examined variability across application months and processing centers using scatter and bar plots.

   2. **Outlier analysis**: Detected unusually long visa processing durations through boxplot analysis, highlighting a small but significant set of extreme delays that heavily influence the overall distribution and summary statistics.

   3. **Correlation analysis**: Analyzed linear relationships between Processing_Days and key numerical variables such as Application_Month and Applicant_Age using correlation matrices and heatmaps. Assessed the presence of seasonality and demographic influence on processing duration.

   4. **Processing center performance analysis**: Grouped applications by Processing_Center and calculated average processing times to compare operational efficiency and identify moderate variations across centers.

   **Feature engineering**: Created informative features to enhance downstream modeling, including:

   1. Application_Month (derived temporal feature),

   2. Peak_Season indicator distinguishing peak and off-peak application periods,

   3. Country-specific average processing time,

   4. Visa-type-wise average processing time,

   5. Visa-class-wise average processing time,

   6. Delay_Status categorical label based on processing time thresholds (Fast, Normal, Delayed, Severely Delayed).


**Milestone 3: Predictive Modeling**

**Objective:** Develop, evaluate, and optimize regression models for accurate visa processing time prediction.

**Tasks Performed:**

1. **Model development:** Implemented baseline regression models including Linear Regression and Random Forest Regressor using the processed feature dataset. Ensured an appropriate train–test split to prevent data leakage and maintain fair model evaluation.

2. **Model training:** Trained each regression model on the training dataset after completing preprocessing and feature encoding steps, ensuring consistent feature representation across all models.

3. **Performance evaluation:** Evaluated model performance using standard regression metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to objectively assess prediction accuracy and error distribution.

4. **Model comparison and selection:** Compared baseline models based on evaluation metrics and overall generalization performance, and selected the best-performing model considering both accuracy and robustness.

5. **Model fine-tuning:** Applied hyperparameter tuning techniques to the selected model to further improve performance, reduce prediction error, and enhance generalization on unseen data.


