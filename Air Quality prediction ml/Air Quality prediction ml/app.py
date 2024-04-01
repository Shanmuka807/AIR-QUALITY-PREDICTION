from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('Air Quality prediction ml\model.joblib')
label_encoder = joblib.load('Air Quality prediction ml\label_encoder.joblib')

@app.route('/')
def home():
    return render_template('Air Quality prediction ml\\templates\index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    soi = float(request.form['soi'])
    noi = float(request.form['noi'])
    rpi = float(request.form['Rpi'])
    spmi = float(request.form['SPMi'])

    # Make a prediction
    prediction = model.predict([[soi, noi, rpi, spmi]])[0]
    predicted_class = label_encoder.inverse_transform([prediction])[0]



if __name__ == '__main__':
    app.run(debug=True)
