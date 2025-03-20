import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

dataset_path = "D:/KUN/Hackathon/EEG/emotions.csv"
df = pd.read_csv(dataset_path)

X = df.drop(columns=['label']) 
y = df['label']  

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

num_features = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], 1, num_features))
X_test = X_test.reshape((X_test.shape[0], 1, num_features))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, num_features)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

model.save("depression_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy:.2%}")

import joblib
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

model = tf.keras.models.load_model("depression_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

df = pd.read_csv("D:/KUN/Hackathon/EEG/emotions.csv")

X = df.drop(columns=['label'])
y = df['label']

label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

rf = RandomForestClassifier()
rf.fit(X, y_encoded)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(10).index.tolist()

def get_user_input():
    print("\nEnter EEG readings below")
    user_input = []
    
    for feature in important_features:
        while True:
            try:
                value = float(input(f"Enter value for {feature}: ").replace(",", ""))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number")
    
    return np.array(user_input).reshape(1, -1)

sample_input = get_user_input()
sample_input_scaled = scaler.transform(sample_input).reshape(1, 1, 10)

prediction = model.predict(sample_input_scaled)
predicted_label = np.argmax(prediction)
predicted_emotion = label_encoder.inverse_transform([predicted_label])

print("\nDepression Prediction Result")
print(f"Predicted Emotion: {predicted_emotion[0]}")
