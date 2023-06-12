from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained Random Forest Classifier model
with open('random_forest_classifier.pkl', 'rb') as f:
    rf_classifier = pickle.load(f)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])

        # Make prediction using the Random Forest Classifier model
        classification = predict_rainfall(temperature, humidity, pressure)

        return render_template('index.html', classification=classification)
    
    return render_template('index.html')

# Function to make prediction using the Random Forest Classifier model
def predict_rainfall(temperature, humidity, pressure):
    # Create new data for prediction
    new_data = pd.DataFrame([[temperature, humidity, pressure]])

    # Make prediction using the Random Forest Classifier model
    classification = rf_classifier.predict(new_data)

    if classification[0] == 1:
        return "Rain is likely."
    else:
        return "No rain is likely."

if __name__ == '__main__':
    app.run(debug=True)
