import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv('cabdata.csv')

# Convert categorical data into numerical data
label_encoder = LabelEncoder()
df['model'] = label_encoder.fit_transform(df['model'])
df['day_of_week'] = label_encoder.fit_transform(df['day_of_week'])
df['Time_Category'] = label_encoder.fit_transform(df['Time_Category'])

# Select features and target variable
features = ['month', 'day_of_week', 'passenger_count', 'model', 'Time_Category', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
target = 'fare_amount'

X = df[features]
y = df[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model with training data
model.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
