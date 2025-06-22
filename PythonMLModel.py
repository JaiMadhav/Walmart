import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('CustomerReturn_with_FraudScore_and_Blacklist(in).csv',low_memory=False)

print(data['Blacklist'].value_counts(dropna=False))
data = data.dropna(subset=['Blacklist'])
print(data['Blacklist'].value_counts(dropna=False))
print(data.head())

data = data.fillna(0)

X = data.drop(columns=['Blacklist', 'Customer_ID','City_Tier',
                       'Total_Orders','Total_Returns','Tenure','Normalized Tenure',
                       'Recency','City_Tier','Return_Recency'])  # Features
y = data['Blacklist']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
