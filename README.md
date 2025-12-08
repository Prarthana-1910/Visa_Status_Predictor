# AI Enabled Visa Status Prediction and Processing time Estimator
**Milestone 1:** Data Collection & Preprocessing 

  **Objective:** Build structured, cleaned dataset for modeling.
  
  **Tasks Performed:**
  1. **Data collection**: Searched multiple reliable websites and downloaded historical records related to visa processing and merged those sources into a single, consistent             dataset (aligned columns, removed duplicates). The target label (Processing_Days) was generated during the creation of dataset.
  2. **Handled missing values**: Identified which columns had missing data and how much was missing and applied appropiate strategies. For columns with numerical datatype, the           missing rows were filled with median and for columns with string datatype, the missing rows were filled with "Unkown" or mode value.
  3. **Encoded categorical features**: The text categorical data was converted into numeric form using one hot encoding, so that models can use them properly and also ensured that       encodings didn’t introduce unintended ordinal relationships where none exist.
