from flask import Flask, jsonify, request
import pandas as pd
import numpy as np

import joblib



model = joblib.load("loan_model.pkl")

app = Flask(__name__)

@app.route('/test', methods =['GET'])

def show():
   return 'server is running good'


@app.route("/predict", methods= ["POST"])

def get_predictions():
    try:
      
      data = request.get_json() 
      
      inputed_data =pd.DataFrame([data], columns=[
      'purpose', 'int.rate', 'installment', 'log.annual.inc',
      'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
      'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'
]) 
      predictions = model.predict(inputed_data)
      
      prediction = np.array(predictions).tolist()
      
      return jsonify({'prediction': prediction[0]})
      
    except Exception as e:
       return jsonify({f"the error -- {e}"}), 400
    




if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)