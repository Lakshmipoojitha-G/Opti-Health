from flask import Flask, jsonify, request
import sys
sys.path.append("./Models")
from Models.symptom_prediction import predictDisease
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Get prediction for given Symptoms.
# Symptoms format : "sym1,sym2,sym3" 
@app.route('/predict-symptoms', methods=['POST'])
def predictSymptoms():
    data = request.get_json()
    # symptoms = "Itching,Skin Rash,Nodal Skin Eruptions"
    symptoms = data["symptoms"]
    res = predictDisease(symptoms)
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True)
