import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 

# Load the data
data = pd.read_csv("C:/Users/kumar/Downloads/LoanApprovalPrediction.csv")

# Display the first few rows
print(data.head())

# Identify categorical variables
obj = (data.dtypes == 'object') 
print("Categorical variables before dropping Loan_ID:", len(list(obj[obj].index)))

# Dropping Loan_ID column if it exists
if 'Loan_ID' in data.columns:
    data.drop(['Loan_ID'], axis=1, inplace=True)

# Recheck categorical variables after dropping
obj = (data.dtypes == 'object') 
object_cols = list(obj[obj].index) 
print("Categorical variables after dropping Loan_ID:", object_cols)

# Plotting categorical distributions
num_cols = len(object_cols)
plt.figure(figsize=(18, (num_cols // 4 + 1) * 5))  # Adjust height based on number of columns

for index, col in enumerate(object_cols, start=1): 
    y = data[col].value_counts()  # Count unique values in each categorical column
    plt.subplot((num_cols // 4) + 1, 4, index)  # Create subplot grid
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    
    if not y.empty:  # Check if y has data
        sns.barplot(x=list(y.index), y=y)  # Create bar plot
        plt.title(col)  # Title for each subplot
    else:
        plt.title(f"No data for {col}")  # Title indicating no data

plt.tight_layout()  # Adjust layout for better spacing
plt.show()  # Display the plots

# Label encoding categorical variables
label_encoder = preprocessing.LabelEncoder() 
for col in object_cols: 
    data[col] = label_encoder.fit_transform(data[col])

# Check again for categorical variables
obj = (data.dtypes == 'object') 
print("Categorical variables after encoding:", len(list(obj[obj].index)))

# Correlation heatmap
plt.figure(figsize=(12, 6)) 
sns.heatmap(data.corr(), cmap='BrBG', fmt='.2f', linewidths=2, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Visualizing gender and marital status by loan status
sns.catplot(x="Gender", y="Married", hue="Loan_Status", kind="bar", data=data)
plt.title('Gender vs Married Status by Loan Status')
plt.show()

# Fill missing values with the mean (only for numerical columns)
for col in data.columns: 
    if data[col].dtype in [np.float64, np.int64]:  # Check for numeric columns
        data[col] = data[col].fillna(data[col].mean())  

print("Remaining missing values per column after filling:", data.isna().sum())  # Check for remaining missing values

# Split the dataset into features and target variable
X = data.drop(['Loan_Status'], axis=1) 
Y = data['Loan_Status'] 
print("Features shape:", X.shape, "Target shape:", Y.shape) 

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1) 
print("Training set shape:", X_train.shape, "Testing set shape:", X_test.shape)

# Classifier instantiation
knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7) 
svc = SVC() 
lc = LogisticRegression(max_iter=300)  # Increased max_iter to help with convergence

# Making predictions on the training set
for clf in (rfc, knn, svc, lc): 
    clf.fit(X_train, Y_train) 
    Y_pred_train = clf.predict(X_train) 
    print("Training Accuracy score of ", clf.__class__.__name__, "=", 100 * metrics.accuracy_score(Y_train, Y_pred_train))

# Making predictions on the testing set
for clf in (rfc, knn, svc, lc): 
    Y_pred_test = clf.predict(X_test) 
    print("Testing Accuracy score of ", clf.__class__.__name__, "=", 100 * metrics.accuracy_score(Y_test, Y_pred_test))
