import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df=pd.read_csv(r"data\features.csv")

#Defining features and target
cat_features=["Peak_Season","Delay_Status"]
num_features=[col for col in df.columns if col not in cat_features+["Processing_Days"]]

X=df[cat_features+num_features]
y=df["Processing_Days"]
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

#Preprocessing
preprocessor=ColumnTransformer(
    transformers=[
        ("OneHot",OneHotEncoder(handle_unknown="ignore"),cat_features),
        ("num",SimpleImputer(strategy="mean"),num_features)
    ]
)

#Define models with Pipelines
models={
    "Linear Regression": Pipeline([
        ("preprocessor",preprocessor),
        ("model",LinearRegression())
    ]),
    "Random Forest Regressor": Pipeline([
        ("preprocesor",preprocessor),
        ("model",RandomForestRegressor(random_state=42))
    ]),
    "Gradient Boosting regressor": Pipeline([
        ("preprocessor",preprocessor),
        ("model",GradientBoostingRegressor(random_state=42))
    ])
}

for name,model_pipeline in models.items():
    model_pipeline.fit(X_train,y_train)
    joblib.dump(model_pipeline,f"models/{name.replace(' ','_')}_pipeline.joblib")
    print(f"{name} trained and saved!")

#Hyperparameter tuning
param_grid={
    "model__n_estimators":[100, 200, 300],
    "model__learning_rate":[0.05, 0.1, 0.2],
    "model__max_depth":[3, 4, 5],
    "model__subsample":[0.8, 1.0],
    "model__min_samples_leaf":[1, 3, 5]
}

gbr_pipeline=Pipeline([
    ("preprocessor",preprocessor),
    ("model",GradientBoostingRegressor(random_state=42))
])

grid_search=GridSearchCV(
    estimator=gbr_pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train,y_train)

best_gbr_pipeline=grid_search.best_estimator_
joblib.dump(best_gbr_pipeline,"models/gradient_boost_pipeline_tuned.joblib")
joblib.dump(grid_search.best_params_,"models/gradient_boost_params.joblib")
print("Best hyperparameters found:")
print(grid_search.best_params_)