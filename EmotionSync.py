import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ✅ Step 1: Load EEG Dataset
dataset_path = "D:/KUN/Hackathon/EEG/emotions.csv"  # Make sure the path is correct
df = pd.read_csv(dataset_path)

# ✅ Step 2: Prepare Features and Target Variable
X = df.drop(columns=['label'])  # Remove target column
y = df['label']  # Target column (POSITIVE, NEGATIVE, NEUTRAL)

# ✅ Step 3: Convert Categorical Labels to Numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert text labels to 0,1,2

# ✅ Step 4: Normalize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize EEG values

# ✅ Step 5: Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# ✅ Step 6: Reshape for LSTM (samples, time steps, features)
num_features = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], 1, num_features))
X_test = X_test.reshape((X_test.shape[0], 1, num_features))

# ✅ Step 7: Build High-Accuracy LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, num_features)),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes (POSITIVE, NEGATIVE, NEUTRAL)
])

# ✅ Step 8: Compile the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ✅ Step 9: Train the Model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# ✅ Step 10: Save the Model and Scaler
model.save("depression_model.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")  # Save label encoder for decoding predictions

# ✅ Step 11: Evaluate Model Performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Model Training Complete! Accuracy: {accuracy:.2%}")






import numpy as np
import joblib
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ✅ Load trained model, scaler, and label encoder
model = tf.keras.models.load_model("depression_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ✅ Load EEG Dataset to Find Important Features
dataset_path = "D:/KUN/Hackathon/EEG/emotions.csv"  # Update with your dataset path
df = pd.read_csv(dataset_path)

X = df.drop(columns=['label'])  # Remove target column
y = df['label']

# Convert labels to numbers
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)

# Train a quick RandomForest model to find important features
rf = RandomForestClassifier()
rf.fit(X, y_encoded)

# Get top 10 important features
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(10).index.tolist()  # Select top 10

# ✅ Function to Get User Input for Only Top 10 Features
def get_user_input():
    print("\n🔹 Enter EEG readings below (Supports scientific notation e.g., 4.62E+00) 🔹")
    user_input = []
    
    for feature in important_features:
        while True:
            try:
                value = float(input(f"Enter value for {feature}: ").replace(",", ""))  # Handle scientific notation
                user_input.append(value)
                break  # Exit loop if input is valid
            except ValueError:
                print("❌ Invalid input. Please enter a valid number (e.g., 4.62 or 4.62E+00)")

    return np.array(user_input).reshape(1, -1)

# ✅ Get user input for only the selected top 10 features
sample_input = get_user_input()

# ✅ Normalize input using saved scaler
sample_input_scaled = scaler.transform(sample_input).reshape(1, 1, 10)  # Reshape for LSTM

# ✅ Make prediction
prediction = model.predict(sample_input_scaled)
predicted_label = np.argmax(prediction)  # Get class with highest probability
predicted_emotion = label_encoder.inverse_transform([predicted_label])  # Convert back to text label

# ✅ Show output
print("\n🧠 **Depression Prediction Result** 🧠")
print(f"🔹 Predicted Emotion: **{predicted_emotion[0]}**")


