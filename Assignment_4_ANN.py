# Import necessary libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Breast Cancer dataset
data = load_breast_cancer()

# Step 2: Convert to a DataFrame for better analysis
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Step 3: Split the dataset into features and target
X = df.drop('target', axis=1)  # Features
y = df['target']              # Target

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the shapes of the processed datasets
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# Import necessary library for feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Step 1: Initialize SelectKBest
# Use f_classif (ANOVA F-statistic) as the score function for classification tasks
k = 10  # Select the top 10 features
selector = SelectKBest(score_func=f_classif, k=k)

# Step 2: Fit the selector to the training data
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Step 3: Get the selected feature indices
selected_features = selector.get_support(indices=True)

# Print the selected feature names
print("Selected Features:")
for i in selected_features:
    print(f"{i}: {data.feature_names[i]}")

# Print the shapes of the transformed datasets
print(f"X_train_selected shape: {X_train_selected.shape}")
print(f"X_test_selected shape: {X_test_selected.shape}")


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the ANN model with chosen parameters
ann_model = MLPClassifier(
    hidden_layer_sizes=(100,),  # Single hidden layer with 100 neurons
    activation='relu',          # ReLU activation function
    solver='adam',              # Adam optimizer
    alpha=0.0001,               # Regularization term
    learning_rate='adaptive',   # Adaptive learning rate
    max_iter=1000,              # Maximum iterations
    random_state=42             # Reproducibility
)

# Fit the ANN model to the training data
ann_model.fit(X_train_selected, y_train)

# Print the training status
print("Model training complete.")

# Predict on the training and test datasets
y_train_pred = ann_model.predict(X_train_selected)
y_test_pred = ann_model.predict(X_test_selected)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print evaluation results
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Classification Report
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix(y_test, y_test_pred))



# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

# Load and preprocess the dataset

def load_and_preprocess_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select top 10 features
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = selector.get_support(indices=True)

    return X_selected, y, df, data, selected_features

# Load data
X_selected, y, df, data, selected_features = load_and_preprocess_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the ANN model

def train_model():
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                          alpha=0.0001, learning_rate='adaptive', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Streamlit App
st.title("Breast Cancer Prediction App")
st.sidebar.header("User Input Features")

# User input for prediction
user_input = {}
for feature in selected_features:
    feature_name = data.feature_names[feature]
    user_input[feature_name] = st.sidebar.slider(
        feature_name, float(df[feature_name].min()), float(df[feature_name].max()), float(df[feature_name].mean())
    )

# Convert user input into a DataFrame
user_data = pd.DataFrame([list(user_input.values())], columns=[data.feature_names[feature] for feature in selected_features])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(user_data)
    prediction_prob = model.predict_proba(user_data)

    st.write("### Prediction Result:")
    if prediction[0] == 1:
        st.write("**The model predicts: Malignant (Cancerous)**")
    else:
        st.write("**The model predicts: Benign (Non-Cancerous)**")

    st.write("### Prediction Probability:")
    st.write(f"Probability of Benign: {prediction_prob[0][0]:.2f}")
    st.write(f"Probability of Malignant: {prediction_prob[0][1]:.2f}")

# Dataset exploration
if st.checkbox("Show Dataset"):
    st.write("### Breast Cancer Dataset")
    st.write(df)

# Model performance
st.sidebar.header("Model Performance")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

st.sidebar.write(f"Training Accuracy: {train_accuracy:.2f}")
st.sidebar.write(f"Testing Accuracy: {test_accuracy:.2f}")


