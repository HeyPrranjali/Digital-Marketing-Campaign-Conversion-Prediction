import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

# Example: Load your dataset (Assume df is your pandas DataFrame)
cust_conv= pd.read_csv("digital_marketing_campaign_dataset.csv" )

# Columns to transform using OneHotEncoding
encoding_cols = ['Gender', 'CampaignChannel', 'CampaignType']

# Use pd.get_dummies to encode categorical columns
encoded_data_df = pd.get_dummies(cust_conv, columns=encoding_cols, drop_first=True)

# Display the transformed data
#print(encoded_data_df.head())


#get the independendent features
X = encoded_data_df.drop(["Conversion","AdvertisingPlatform","AdvertisingTool"] ,axis=1) 
#get the dependent feature
y = encoded_data_df["Conversion"] 


# Step 1: Train the XGBClassifier model on all features
model = XGBClassifier()
model.fit(X , y)

# Step 2: Get feature importances from the trained model
importances = model.feature_importances_

# Step 3: Identify the top 5 features based on importance
top_features_idx = np.argsort(importances)[::-1][:5]  # Get indices of top 5 features
top_features = X.columns[top_features_idx]

# Step 4: Select only the top 5 features
X_top5 = X[top_features]

# Step 5: Retrain the model using only the top 5 features
model_top5 = XGBClassifier()
model_top5.fit(X_top5, y)

# Step 6: Save the model using joblib
joblib.dump(model_top5, 'xgb_model_top5_features.joblib')



