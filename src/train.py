import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer


df=pd.read_csv(r"data\features.csv")
df_encoded=pd.get_dummies(df,columns=["Peak_Season","Delay_Status"])

X=df_encoded.drop(columns=["Processing_Days"])
y=df_encoded["Processing_Days"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

imputer=SimpleImputer(strategy='mean')
X_train=imputer.fit_transform(X_train)
X_test=imputer.transform(X_test)

models={
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting regressor": GradientBoostingRegressor(random_state=42)
}
results={}
for name,model in models.items():
    model.fit(X_train,y_train)
    joblib.dump(model,f"models/{name}.joblib")

joblib.dump(imputer,"models/imputer.joblib")

#Hyperparameter tuning
param_grid={
    "n_estimators":[100, 200, 300],
    "learning_rate":[0.05, 0.1, 0.2],
    "max_depth":[3, 4, 5],
    "subsample":[0.8, 1.0],
    "min_samples_leaf":[1, 3, 5]
}
gbr=GradientBoostingRegressor(random_state=42)
grid_search=GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train,y_train)
best_gbr=grid_search.best_estimator_
joblib.dump(best_gbr,"models/gradient_boost_tuned.joblib")
joblib.dump(grid_search.best_params_,"models/gradient_boost_params.joblib")
print("Best hyperparameters found:")
print(grid_search.best_params_)