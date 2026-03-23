import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import load_walmart_data

#Load dataset
df = load_walmart_data()
print("Dataset preview:")
print(df.head())
print("\nColumns:", df.columns)

#Convert date to datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst = True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns=['Date'], inplace=True)

#Define features (X) and target (y)
target = 'Weekly_Sales'
X = df.drop(columns=[target])
y = df[target]

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)