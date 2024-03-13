import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from io import StringIO
import contextlib

def process_data_and_train_model(file, epochs):
    st.info("Loading and processing data...")
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return

    # Drop rows with missing values
    df.dropna(inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df)

    # Split data into features and target
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build a simpler neural network
    model = Sequential([
        Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    st.info("Training model...")
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Save model architecture summary to txt file
    st.info("Saving model architecture summary...")
    with open('architecture_summary.txt', 'w') as file:
        with contextlib.redirect_stdout(file):
            model.summary()

    # Display training history
    st.subheader("Training History")
    st.line_chart(pd.DataFrame(history.history))

    st.success("Model trained. Model architecture summary saved as 'architecture_summary.txt'.")

def main():
    st.title("Future Application")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        epochs = st.number_input("Enter number of epochs for training:", min_value=1, value=10)
        if st.button("Train Model"):
            process_data_and_train_model(uploaded_file, epochs)

if __name__ == "__main__":
    main()
