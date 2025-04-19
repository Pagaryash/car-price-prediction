from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load models and scalers
rforest_model = joblib.load("rforest_model.pkl")
lin_reg = joblib.load("lin_reg.pkl")
ss_x = joblib.load("scaler_x.pkl")
ss_y = joblib.load("scaler_y.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['popularity']),
        float(request.form['year']),
        float(request.form['engine_hp']),
        float(request.form['engine_cylinders']),
        float(request.form['highway_mpg']),
        float(request.form['city_mpg'])
    ]
    input_scaled = ss_x.transform([features])
    
    pred_rf_scaled = rforest_model.predict(input_scaled)
    pred_lr_scaled = lin_reg.predict(input_scaled)
    
    pred_rf = ss_y.inverse_transform(pred_rf_scaled.reshape(-1, 1))[0][0]
    pred_lr = ss_y.inverse_transform(pred_lr_scaled.reshape(-1, 1))[0][0]
    
    return render_template('index.html', pred_rf=round(pred_rf, 2), pred_lr=round(pred_lr, 2))

if __name__ == '__main__':
    app.run(debug=True)

