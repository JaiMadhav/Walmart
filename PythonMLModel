'''import pandas as pd

# Load your dataset
df = pd.read_csv('CustomerReturnDataset(in).csv', low_memory=False)  

# Calculate fraud score and update Blacklist directly
df['fraud_score'] = (
    10 * df['Return_Frequency'] +
     8 * df['High_Value_Item_Abuse'] +
     8 * df['Return_Window_Abuse'] +
     8 * df['Return_Reason_Vagueness_Score'] +
     6 * (df['Avg_Order_Value'] / 2807.22) +
     8 * (df['Avg_Return_Value'] / 1337.48) +
     6 * (1 - df['Renormalized Tenure']) +
     6 * df['Normalized Recency'] +
     7 * df['Normalized RR'] +
     7 * df['Repeated_Return_Items'] +
     7 * df['Return_Category_Count']
)

# Apply logic to update Blacklist
cutoff = df['fraud_score'].quantile(0.55)
df['Blacklist'] = (df['fraud_score'] >= cutoff).astype(int)
df.to_csv('CustomerReturn_with_FraudScore_and_Blacklist.csv', index=False)

print("Max fraud score:", df['fraud_score'].max())
print("Cutoff value:", cutoff)
print("Number of blacklisted users:", df['Blacklist'].sum())

'''
'''
import csv

# Read and modify the CSV file
input_file = 'CustomerReturnDataset(in).csv'
output_file = 'modified_data.csv'

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
    # Write the header to the output file
    writer.writeheader()
    
    for row in reader:
        # Apply the condition: Increase salary by 10% if age > 30
        if float(row['Return_Frequency']) > 0.3:
            row['Blacklist'] = 1
        else:
            row['Blacklist'] = 0
        writer.writerow(row)

print(f"Modified CSV file saved as '{output_file}'.")
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Step 1: Load the dataset
# Replace 'your_file.csv' with the path to your CSV file
data = pd.read_csv('CustomerReturn_with_FraudScore_and_Blacklist(in).csv',low_memory=False)

print(data['Blacklist'].value_counts(dropna=False))
data = data.dropna(subset=['Blacklist'])
print(data['Blacklist'].value_counts(dropna=False))
# Step 2: Explore the data (optional)
print(data.head())

data = data.fillna(0)

# Step 3: Define features (X) and target (y)
# Replace 'Outcome' with the name of your target column
X = data.drop(columns=['Blacklist', 'Customer_ID','City_Tier',
                       'Total_Orders','Total_Returns','Tenure','Normalized Tenure',
                       'Recency','City_Tier','Return_Recency'])  # Features
y = data['Blacklist']  # Target


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 5: Initialize and train the logistic regression model
model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Predict for new data (optional)
# Replace 'new_data' with your new input data as a DataFrame
# new_data = pd.DataFrame({...})
# predictions = model.predict(new_data)
# print(predictions)
'''
new_data = pd.DataFrame([{
    'Total_Orders': 12,
    'Total_Returns': 12,
    'Return_Frequency': 1,
    'Avg_Order_Value': 2200,
    'Avg_Return_Value': 2100,
    'Tenure': 15,
    'Recency': 3,
    'Repeated_Return_Items': 2,
    'Return_Category_Count': 4,
    'Return_Recency': 2,
    'Return_Window_Abuse': 1,
    'Return_Reason_Vagueness_Score': 0.6,
    'Return_Spike_Post_Sale': 1,
    'High_Value_Item_Abuse': 1
}])
'''
print(y.value_counts())
print("Predictions:", y_pred[:100])  # check first 20 predictions
print("Actual     :", y_test.values[:100])  # compare with actual values
y_proba = model.predict_proba(X_test)

'''
# Step 9: Scale new data and predict
new_prediction = model.predict(new_data)
print(f"Predicted Blacklist Status: {new_prediction[0]}")
'''
