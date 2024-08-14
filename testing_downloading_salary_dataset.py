from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features # everyother category listed e.g. ID, age, workclas, ..., native-country etc.
y = adult.data.targets # income

print("adult.metadata:\n")
  
# metadata 
print(adult.metadata) 

print("\nadult.variables:\n")
  
# variable information 
print(adult.variables) 

print("\nX:\n")

print(X)

print("\ny:\n")

print(y)

print("\n\n\n")

# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

print("students.metadata:\n")

# metadata 
print(predict_students_dropout_and_academic_success.metadata) 

print("\nstudents.variables:\n")
  
# variable information 
print(predict_students_dropout_and_academic_success.variables) 

print("\nX:\n")

print(X)

print("\ny:\n")

print(y) # dropout, graduate, enrolled

import pandas as pd

# Combine features and targets into a single DataFrame
df = pd.concat([predict_students_dropout_and_academic_success.data.features, predict_students_dropout_and_academic_success.data.targets], axis=1)

# Print out the column labels of the dataset
print("Column labels:\n", df.columns)

print(f"\n\ndataset types:\n {df.dtypes}")

# If there are any categorical columns (dtype == 'object' or 'category'), perform encoding
if df.select_dtypes(include=['object', 'category']).shape[1] > 0:
    df = pd.get_dummies(df)

features = df.drop(columns=['Target_Dropout', 'Target_Enrolled', 'Target_Graduate', 'Gender', 'Nacionality', 'Educational special needs']).values  # Exclude target and protected attribute
labels = df[['Target_Dropout', 'Target_Enrolled', 'Target_Graduate']].values  # Example: using 'Dropout' as labels

protected_attr = df[['Gender', 'Nacionality', 'Educational special needs']].values
print(f"\n\nprotected attributes:\n{protected_attr}\n\n and its length:\n{len(protected_attr)}")

from imblearn.over_sampling import ADASYN  # Alternative to SMOTE for imbalanced datasets
from sklearn.utils import resample # to resample protected attributes appropriately
from sklearn.preprocessing import StandardScaler  # Module for data normalization
import torch
device = 'cpu'
adasyn = ADASYN()
features, labels = adasyn.fit_resample(features, labels)

# Resample protected attributes accordingly
# protected_attr_resampled = np.repeat(protected_attr, labels.shape[0] // len(protected_attr), axis=0)
# protected_attr_resampled = protected_attr
protected_attr_resampled = resample(protected_attr, n_samples=labels.shape[0], random_state=42)

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convert features, labels, and protected attribute to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)  # Use float32 for multi-label
protected_attr_tensor = torch.tensor(protected_attr_resampled, dtype=torch.float32).to(device)

# Check sizes of tensors
print(f"Features tensor size: {features_tensor.size()}")
print(f"Labels tensor size: {labels_tensor.size()}")
print(f"Protected attribute tensor size: {protected_attr_tensor.size()}")
print(f"\n\nprotected attributes:\n{protected_attr_tensor[-1050:-750]}\n\n and its length:\n{len(protected_attr_tensor)}")

# # Extract second and third columns
# second_col = protected_attr_tensor[:, 1].numpy()
# third_col = protected_attr_tensor[:, 2].numpy()

# # Combine these columns into a DataFrame for easier manipulation
# df_protected = pd.DataFrame({'Column2': second_col, 'Column3': third_col})

# # Count the occurrences of each unique combination
# distribution = df_protected.value_counts().reset_index(name='Count')

# print("Protected attributes distribution in data set:")
# print(distribution)
