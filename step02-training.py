import yfinance as yf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Same preparatory code as in step01-labeling.py
ticker = "KO"
today = datetime.now().strftime('%Y-%m-%d')
data = yf.Ticker(ticker).history(start="2007-01-01", end=today)
df = data[["Open", "High", "Low", "Close"]].copy()
df['Range'] = df['High'] - df['Low']
df['Body'] = abs(df['Close'] - df['Open'])
df['BodyRangeRatio'] = df['Body'] / df['Range']
threshold = 0.001
df['Direction'] = df.apply(
    lambda row: 2 if abs(row['Open'] - row['Close']) < threshold else (1 if row['Close'] > row['Open'] else 0),
    axis=1
)
df['CoM'] = (df[['Open', 'Close']].mean(axis=1) - df['Low']) / df['Range'] 
df['UpperWick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Range']
df['LowerWick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Range']

# Pull training data from CSV
train_df = pd.read_csv(f"{ticker}-training.csv")

# Define features to include in the training set
features = [
    'BodyRangeRatio',
    'CoM',
    'Direction',
    'UpperWick',
    'LowerWick'
]

# Training Data Set
X = train_df[features]
# Labeled vector
y = train_df['Label']

# Encode training labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split X into testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

# Create and fit the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Compute metrics and classification report, including F1 Score, Precision, and Recall rates
y_pred = model.predict(X_test)
target_names = [str(cls) for cls in encoder.classes_]
print(classification_report(y_test, y_pred, labels=range(len(encoder.classes_)), target_names=target_names))

# Visualize the feature importances
importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Feature Importances")
plt.show()

# Save model as pickle file to upload in other contexts
def save_model(model, encoder, filename):
    joblib.dump(model, filename + '_model.pkl')
    # joblib.dump(encoder, filename + '_encoder.pkl')

save_model(model, encoder, f"{ticker}-model")