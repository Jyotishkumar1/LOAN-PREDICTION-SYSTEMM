import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("C:/Users/kumar/Downloads/Loan-prediction-using-Machine-Learning-and-Python-master/loan_data.csv")

# Display initial information
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# Visualize Loan Status distribution
temp = df['Loan_Status'].value_counts()
plt.pie(temp.values, labels=temp.index, autopct='%1.1f%%')
plt.title("Loan Status Distribution")
plt.show()

# Visualize categorical features
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['Gender', 'Married']):
    plt.subplot(1, 2, i+1)
    sb.countplot(data=df, x=col, hue='Loan_Status')
plt.tight_layout()
plt.title("Count Plot for Gender and Married Status")
plt.show()

# Visualize distributions of numeric features
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.histplot(df[col], kde=True)  # Using histplot instead of distplot for newer versions
plt.tight_layout()
plt.title("Distribution of Applicant Income and Loan Amount")
plt.show()

# Visualize boxplots for numeric features
plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome','LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.title("Boxplot of Applicant Income and Loan Amount")
plt.show()

# Filter out outliers
df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]

# Mean Loan Amount by Gender
print(df.groupby('Gender').mean(numeric_only=True)['LoanAmount'])

# Mean Loan Amount by Marriage status and Gender
print(df.groupby(['Married', 'Gender']).mean(numeric_only=True)['LoanAmount'])

# Function to apply label encoding
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

# Applying function to entire DataFrame
df = encode_labels(df)

# Generating Heatmap
plt.figure(figsize=(10, 8))
sb.heatmap(df.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title("Heatmap of Correlations")
plt.show()

# Define features and target variable
features = df.drop('Loan_Status', axis=1)
target = df['Loan_Status'].values

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)

# Balancing the data using RandomOverSampler
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

# Normalizing the features for stable and fast training
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val = scaler.transform(X_val)

# Train the SVM model
model = SVC(kernel='rbf', probability=True)  # Set probability=True for ROC AUC
model.fit(X_resampled, Y_resampled)

# Evaluate the model
training_predictions = model.predict(X_resampled)
validation_predictions = model.predict(X_val)

print('Training ROC AUC Score:', roc_auc_score(Y_resampled, training_predictions))
print('Validation ROC AUC Score:', roc_auc_score(Y_val, validation_predictions))
print()

# Confusion matrix
cm = confusion_matrix(Y_val, validation_predictions)

plt.figure(figsize=(6, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification report
print(classification_report(Y_val, validation_predictions))
