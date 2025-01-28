# Flask App for Downtime Prediction

This is a Flask-based web application that uses a logistic regression model to predict whether downtime will occur based on "Temperature" and "Run Time" data. The app allows users to upload a CSV file with relevant data, train a machine learning model, and make predictions on new inputs.

## Features:
- **Upload CSV**: Upload a CSV file containing "Temperature", "Run_Time", and "Downtime_Flag" columns.
- **Train Model**: Train a logistic regression model using the uploaded data.
- **Predict**: Input "Temperature" and "Run Time" to predict whether downtime will occur based on the trained model.

## Requirements

Before running the app, make sure you have the following installed:

- Python 3.7 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Clone the Repository
Clone the repository to your local machine using Git:

```bash
git clone https://github.com/username/flaskapp.git
cd flask-downtime-prediction

```
### 2.Create a Virtual Environment 
python -m venv venv
On Windows:
    venv\Scripts\activate
On macOS/Linux:
   venv/bin/activate


### 3.Install Dependencies
pip install -r requirements.txt


### 4. Prepare Your Data


### 5. Run the Application
python app.py

### open the browser
http://127.0.0.1:5000/

