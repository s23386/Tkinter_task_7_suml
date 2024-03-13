import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from io import StringIO

def process_data_and_train_model(epochs):
  messagebox.showinfo("Information", "Application launched. Loading and processing data...")

  # Load data
  file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
  if not file_path:
    messagebox.showerror("Error", "No file selected.")
    return

  try:
    df = pd.read_csv(file_path)
  except Exception as e:
    messagebox.showerror("Error", f"Error loading CSV file: {e}")
    return

  # Handle missing values
  for col in df.columns:
    if df[col].isnull().sum() > 0:
      if pd.api.types.is_numeric_dtype(df[col]):
        df[col].fillna(df[col].mean(), inplace=True)  # Replace with preferred method for numeric data
      else:
        # Handle non-numeric data (choose one approach)
        # Option 1: Remove column (if not important)
        # df.drop(col, axis=1, inplace=True)

        # Option 2: Fill with a constant value (e.g., 'NA')
        df[col].fillna('NA', inplace=True)

        # Option 3: Implement a more sophisticated strategy (e.g., label encoding)

  # One-hot encode categorical variables (replace with actual column names)
  categorical_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']  # Replace with your list of categorical columns
  df = pd.get_dummies(df, columns=categorical_cols)

  # Split data into features and target
  X = df.drop(columns=['Survived'])
  y = df['Survived']

  # Split data into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Standardize features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Build a simpler neural network (adjust layers/neurons if needed)
  model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
  ])

  # Compile model
  optimizer = Adam(learning_rate=0.001)  # Adjust learning rate if necessary
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

  # Train model
  try:
    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_data=(X_test_scaled, y_test))
  except Exception as e:
    messagebox.showerror("Error", f"Error during training: {e}")
    return

  # Save information about the network architecture to a .txt file
  with open('architecture_summary.txt', 'w') as file:
    model.summary(print_fn=lambda x: file.write(x))  # Remove line_break argument

  messagebox.showinfo("Information", "Model trained. Model architecture summary saved as 'architecture_summary.txt'.")

def create_window():
  window = tk.Tk()
  window.title("Future Application")
  
  # Set icon for the window (replace with the correct path)
  window.iconbitmap('/Users/smilan/Desktop/PJA/8th Semester/SUML/Task 7./icon.ico')

  label = tk.Label(window, text="Enter number of epochs for training:")
  label.pack(padx=20, pady=5)

  entry = tk.Entry(window)
  entry.pack(padx=20, pady=5)

  button = tk.Button(window, text="Train Model", command=lambda: process_data_and_train_model(int(entry.get())))
  button.pack(padx=20, pady=20)

  window.mainloop()

if __name__ == "__main__":
  create_window()
