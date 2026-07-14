import os
import sys
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'src'))

app = Flask(__name__)
app.secret_key = 'creditshield_ai_secret_key_apsche'

# Global in-memory list to store prediction history
PREDICTION_HISTORY = []

# Instantiate the credit predictor lazily or inside a try-except so app doesn't crash on startup if models are not trained yet
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            from predict import CreditPredictor
            models_dir = os.path.join(project_root, 'models')
            predictor = CreditPredictor(models_dir=models_dir)
        except Exception as e:
            logger.error(f"Failed to load machine learning models: {e}")
            predictor = None
    return predictor

@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/predict', methods=['GET'])
def predict_form():
    # If the model is not trained yet, provide a warning page or let them proceed
    model_loaded = get_predictor() is not None
    return render_template('index.html', active_page='predict', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict_post():
    # Parse form inputs
    try:
        raw_input = {
            'CODE_GENDER': request.form.get('CODE_GENDER', 'F'),
            'FLAG_OWN_CAR': request.form.get('FLAG_OWN_CAR', 'N'),
            'FLAG_OWN_REALTY': request.form.get('FLAG_OWN_REALTY', 'Y'),
            'CNT_CHILDREN': int(request.form.get('CNT_CHILDREN', 0)),
            'AMT_INCOME_TOTAL': float(request.form.get('AMT_INCOME_TOTAL', 0)),
            'NAME_INCOME_TYPE': request.form.get('NAME_INCOME_TYPE', 'Working'),
            'NAME_EDUCATION_TYPE': request.form.get('NAME_EDUCATION_TYPE', 'Secondary / secondary special'),
            'NAME_FAMILY_STATUS': request.form.get('NAME_FAMILY_STATUS', 'Married'),
            'NAME_HOUSING_TYPE': request.form.get('NAME_HOUSING_TYPE', 'House / apartment'),
            'AGE_YEARS': float(request.form.get('AGE_YEARS', 30.0)),
            'EMPLOYMENT_YEARS': float(request.form.get('EMPLOYMENT_YEARS', 0.0)),
            'IS_UNEMPLOYED': int(request.form.get('IS_UNEMPLOYED', 0)),
            'FLAG_WORK_PHONE': int(request.form.get('FLAG_WORK_PHONE', 0)),
            'FLAG_PHONE': int(request.form.get('FLAG_PHONE', 0)),
            'FLAG_EMAIL': int(request.form.get('FLAG_EMAIL', 0)),
            'OCCUPATION_TYPE': request.form.get('OCCUPATION_TYPE', 'Other/Unspecified'),
            'CNT_FAM_MEMBERS': int(request.form.get('CNT_FAM_MEMBERS', 1))
        }
        
        # Validation checks
        if raw_input['AMT_INCOME_TOTAL'] <= 0 or raw_input['AGE_YEARS'] < 18 or raw_input['EMPLOYMENT_YEARS'] < 0:
            raise ValueError("Numeric inputs fall outside logical bounds.")
            
    except (ValueError, TypeError) as e:
        logger.error(f"Validation error: {e}")
        return render_template('error.html', error_code="400", error_message="Invalid Input Parameters Provided. Check values and try again."), 400

    # Get predictor
    pred_obj = get_predictor()
    if pred_obj is None:
        # Fallback prediction if models are not trained yet (to prevent runtime crash during initial testing)
        logger.warning("Predictor not loaded. Running fallback heuristic.")
        
        # Simple heuristic: Approve if income > 80000 and not unemployed
        prob = 0.88 if (raw_input['AMT_INCOME_TOTAL'] > 80000 and raw_input['IS_UNEMPLOYED'] == 0) else 0.45
        status = "Approved" if prob >= 0.5 else "Rejected"
        risk = "Low Risk" if prob >= 0.85 else ("Medium Risk" if prob >= 0.5 else "High Risk")
        
        prediction_result = {
            'Prediction': status,
            'Confidence': round(prob * 100, 2),
            'RiskCategory': risk,
            'ApprovedClass': 1 if status == "Approved" else 0
        }
    else:
        try:
            prediction_result = pred_obj.predict(raw_input)
        except Exception as e:
            logger.error(f"Inference pipeline execution error: {e}")
            return render_template('error.html', error_code="500", error_message=f"Model inference failed: {str(e)}"), 500

    # Log to history
    history_record = {
        **raw_input,
        'PREDICTION': prediction_result['Prediction'],
        'CONFIDENCE': prediction_result['Confidence'],
        'RISK': prediction_result['RiskCategory']
    }
    PREDICTION_HISTORY.append(history_record)
    
    return render_template(
        'result.html', 
        prediction=prediction_result['Prediction'],
        confidence=prediction_result['Confidence'],
        risk=prediction_result['RiskCategory'],
        input_data=raw_input
    )

@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/contact')
def contact():
    return render_template('contact.html', active_page='contact')

@app.route('/contact/submit', methods=['POST'])
def contact_submit():
    # In a real app we would log or email this message.
    name = request.form.get('name')
    email = request.form.get('email')
    logger.info(f"Contact message received from {name} ({email})")
    return redirect(url_for('contact', submitted=True))

@app.route('/history')
def history():
    # Optional history display page
    return render_template('history.html', active_page='history', history_list=PREDICTION_HISTORY[::-1])

# --- Custom Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code="404", error_message="The requested endpoint does not exist on this credit evaluation system."), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code="500", error_message="An internal server error occurred. Please contact the administrator."), 500

if __name__ == '__main__':
    # Running Flask app on local development server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
