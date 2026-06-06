from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

file_path = r"C:\Users\rahul\Desktop\credit scoring project\cleaned_file.csv"

df= pd.read_csv(file_path)
print(df.columns)
y= df['Default']
X= df.drop(columns= ['Default'])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
# print(scaler.feature_names_in_)
# print(X_train[0])
X_test = scaler.transform(X_test)
