import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from prettytable import PrettyTable

df=pd.read_csv("data/features.csv")
df_encoded=pd.get_dummies(df,columns=["Peak_Season","Delay_Status"])

X=df_encoded.drop(columns=["Processing_Days"])
y=df_encoded["Processing_Days"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

imputer=joblib.load("models/imputer.joblib")
X_test=imputer.transform(X_test)

model_files={
    "Linear Regression":"models\Linear Regression.joblib",
    "Random Forest":"models\Random Forest Regressor.joblib",
    "Gradient Boosting":"models\Gradient Boosting regressor.joblib"
}

table=PrettyTable()
table.field_names=["Model","MAE","RMSE","R**2"]
res={}

for name,path in model_files.items():
    model=joblib.load(path)
    y_pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)
    res[name]=mae
    table.add_row([name,round(mae,3),round(rmse,3),round(r2,3)])

print("Model Evaluation Results:")
print(table)

best_model=min(res,key=res.get)
print(f"Best model based on MAE: {best_model}")

#MAE for tuned model
tuned_model=joblib.load("models/gradient_boost_tuned.joblib")
y_pred_tuned=tuned_model.predict(X_test)
mae_tuned=mean_absolute_error(y_test,y_pred_tuned)
print(f"Tuned Gradient Boosting MAE: {mae_tuned:.4f}")