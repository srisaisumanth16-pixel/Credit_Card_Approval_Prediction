import urllib.request
import urllib.parse
import json

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    print("Starting programmatic integration tests for Flask web application...")
    
    # 1. Test Home Page
    try:
        response = urllib.request.urlopen(f"{base_url}/")
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "Dashboard" in html or "CreditShield" in html
        print("[PASS] GET / (Home Dashboard loads correctly)")
    except Exception as e:
        print(f"[FAIL] GET / failed: {e}")
        return False
        
    # 2. Test Predict Form Page
    try:
        response = urllib.request.urlopen(f"{base_url}/predict")
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "Application Form" in html or "predict-form" in html
        print("[PASS] GET /predict (Evaluation form loads correctly)")
    except Exception as e:
        print(f"[FAIL] GET /predict failed: {e}")
        return False
        
    # 3. Test Form Submission (Approved profile prediction flow)
    try:
        # Sample approved profile parameters
        data = {
            'CODE_GENDER': 'F',
            'FLAG_OWN_CAR': 'Y',
            'FLAG_OWN_REALTY': 'Y',
            'CNT_CHILDREN': '0',
            'AMT_INCOME_TOTAL': '180000.0',
            'NAME_INCOME_TYPE': 'Working',
            'NAME_EDUCATION_TYPE': 'Higher education',
            'NAME_FAMILY_STATUS': 'Married',
            'NAME_HOUSING_TYPE': 'House / apartment',
            'AGE_YEARS': '32.5',
            'EMPLOYMENT_YEARS': '5.5',
            'IS_UNEMPLOYED': '0',
            'FLAG_WORK_PHONE': '0',
            'FLAG_PHONE': '1',
            'FLAG_EMAIL': '0',
            'OCCUPATION_TYPE': 'Core staff',
            'CNT_FAM_MEMBERS': '2'
        }
        encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(f"{base_url}/predict", data=encoded_data, method='POST')
        
        response = urllib.request.urlopen(req)
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "Application" in html
        assert "Result" in html or "Approved" in html or "Rejected" in html
        print("[PASS] POST /predict (Single-record model scoring runs successfully)")
    except Exception as e:
        print(f"[FAIL] POST /predict failed: {e}")
        return False
        
    # 4. Test About page
    try:
        response = urllib.request.urlopen(f"{base_url}/about")
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "About Project" in html or "Specifications" in html
        print("[PASS] GET /about (Technical Specs loads correctly)")
    except Exception as e:
        print(f"[FAIL] GET /about failed: {e}")
        return False
        
    # 5. Test Contact page
    try:
        response = urllib.request.urlopen(f"{base_url}/contact")
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "Contact" in html
        print("[PASS] GET /contact (Contact info loads correctly)")
    except Exception as e:
        print(f"[FAIL] GET /contact failed: {e}")
        return False

    # 6. Test Contact Form Submission redirect
    try:
        data = {
            'name': 'Test User',
            'email': 'test@example.com',
            'subject': 'APSCHE Test Query',
            'message': 'Hello, checking server forms.'
        }
        encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(f"{base_url}/contact/submit", data=encoded_data, method='POST')
        
        response = urllib.request.urlopen(req)
        # Check that it redirected back with ?submitted=True
        assert response.status == 200
        assert "submitted=True" in response.url or "submitted=true" in response.url or "Contact" in response.read().decode('utf-8')
        print("[PASS] POST /contact/submit (Contact redirect form triggers correctly)")
    except Exception as e:
        print(f"[FAIL] POST /contact/submit failed: {e}")
        return False
        
    # 7. Test Evaluation History page
    try:
        response = urllib.request.urlopen(f"{base_url}/history")
        html = response.read().decode('utf-8')
        assert response.status == 200
        assert "History" in html
        # Check that our prediction is logged inside history table
        assert "Approved" in html or "Rejected" in html
        print("[PASS] GET /history (Prediction logs and statistics display correctly)")
    except Exception as e:
        print(f"[FAIL] GET /history failed: {e}")
        return False
        
    print("\n>>> ALL 7 FLASK INTEGRATION TESTS COMPLETED SUCCESSFULLY! <<<")
    return True

if __name__ == '__main__':
    test_endpoints()
