import pandas as pd
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load data
df = pd.read_csv(
    r"C:\Users\SARAH ZETA\OneDrive\Documents\deploy project\Churn_Modelling.csv"
    )

X = df[['CreditScore', 'Geography', 'Gender', 'Age', 
        'Tenure', 'Balance', 'NumOfProducts', 
        'IsActiveMember','EstimatedSalary']]

y = df['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state= 42,
    stratify= y
)

# Encoding and Scaling
cat_label = ['Gender']
cat_ohe = ['Geography']
num = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

std = StandardScaler()
label = LabelEncoder()

X_train = pd.get_dummies(X_train, columns=cat_ohe, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_ohe, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

label_train = label.fit_transform(X_train[cat_label])
label_test = label.transform(X_test[cat_label])

X_train[cat_label] = label_train.reshape(-1, 1)
X_test[cat_label] = label_test.reshape(-1, 1)

X_train[num] = std.fit_transform(X_train[num])
X_test[num] = std.transform(X_test[num])

# Train model
model = SVC(class_weight= 'balanced', probability= True)

model.fit(X_train, y_train)

# Save model
os.makedirs('model', exist_ok= True)

joblib.dump(model, 'model/bank_churn_model.pkl')
joblib.dump(std, 'model/standard_scaler.pkl')
joblib.dump(label, 'model/label_encoder.pkl')

print("✅ Model & encoder saved successfully!")
print("Training columns:", X_train.columns.tolist())