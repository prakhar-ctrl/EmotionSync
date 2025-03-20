EEG-Based Emotion Detection Using LSTM
📌 Project Overview
This project uses EEG data to classify emotions (Positive, Negative, Neutral) using a deep learning model built with LSTM networks. A RandomForest classifier helps identify the most important EEG features.

🚀 Features
Data Preprocessing: Label encoding, feature scaling, and train-test split.
LSTM Model: Trained with 128 and 64 LSTM units, dropout layers, and a softmax classifier.
Feature Selection: Uses RandomForest to extract the top 10 most relevant EEG features.
User Input Prediction: Takes real-time EEG readings to predict emotional states.
📂 File Structure
emotions.csv → EEG dataset
depression_model.h5 → Trained LSTM model
scaler.pkl → StandardScaler for normalizing input
label_encoder.pkl → Label encoder for decoding predictions
main.py → Code for training and prediction
🛠 How to Run
Install dependencies:
bash
Copy
Edit
pip install tensorflow pandas numpy scikit-learn joblib
Run the training script:
bash
Copy
Edit
python main.py
Enter EEG readings to predict the emotion.
🔍 Results
The model achieves ~XX% accuracy on test data. Future improvements may include more EEG channels and hyperparameter tuning.

