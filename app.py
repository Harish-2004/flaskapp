from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

app = Flask(__name__)
app.secret_key = "r8724573423823"

# Paths
DATA_PATH = "/tmp/uploaded_data.csv"
MODEL_PATH = "/tmp/model.pkl"

@app.route('/')
def index():
    return render_template('index.html')

# Upload endpoint
@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        flash("No file part", "danger")
        return render_template('index.html')

    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "danger")
        return render_template('index.html')

    # Check if the uploaded file is a CSV file
    if not file.filename.endswith('.csv'):
        flash("Invalid file type. Please upload a CSV file.", "danger")
        return render_template('index.html')

    try:
        # Read the CSV file
        data = pd.read_csv(file)

        # Check for required columns
        required_columns = {"Temperature", "Run_Time", "Downtime_Flag"}
        if not required_columns.issubset(data.columns):
            flash(f"Dataset must contain the following columns: {', '.join(required_columns)}", "danger")
            return render_template('index.html')

        # Save the file to /tmp (Vercel temporary storage)
        file.seek(0)  # Reset file pointer to the beginning
        file.save(DATA_PATH)
        flash("File uploaded successfully!", "success")
        return render_template('index.html')

    except Exception as e:
        flash(f"An error occurred while processing the file: {str(e)}", "danger")
        return render_template('index.html')


# Train endpoint
@app.route('/train', methods=['GET'])
def train_model():
    if not os.path.exists(DATA_PATH):
        flash("No data found. Please upload data first.", "danger")
        return redirect(url_for('index'))

    # Load dataset
    data = pd.read_csv(DATA_PATH)
    if not {"Temperature", "Run_Time", "Downtime_Flag"}.issubset(data.columns):
        flash("Dataset must contain 'Temperature', 'Run_Time', and 'Downtime_Flag' columns.", "danger")
        return redirect(url_for('index'))

    # Prepare data
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model to /tmp (Vercel temporary storage)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return render_template('train.html', accuracy=accuracy, f1_score=f1)

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        flash("No trained model found. Please train the model first.", "danger")
        return redirect(url_for('index'))

    # Load input data
    temperature = request.form.get("Temperature", type=float)
    run_time = request.form.get("Run_Time", type=float)

    if temperature is None or run_time is None:
        flash("Invalid input. Please provide both Temperature and Run Time.", "danger")
        return redirect(url_for('index'))

    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # Make prediction
    X = [[temperature, run_time]]
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    return render_template('predict.html', downtime="Yes" if prediction == 1 else "No", confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
