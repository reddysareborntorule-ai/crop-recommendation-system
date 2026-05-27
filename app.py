from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
from utils.config import *

app = Flask(__name__)

# Load model and encoders
def load_model_and_encoders():
    """Load trained XGBoost model and encoders"""
    try:
        with open('models/xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        print("Model and encoders loaded successfully")
        return model, encoders
    except FileNotFoundError:
        print("Model files not found. Please train the model first.")
        return None, None

# Initialize model and encoders
model, encoders = load_model_and_encoders()

if model is None or encoders is None:
    # Create dummy model for development (remove in production)
    print("Running in development mode with dummy model")
    model = None
    encoders = {
        'district': None,
        'season': None,
        'soil': None,
        'crop': None
    }

# Load configuration
from utils.config import *

def calculate_fertilizer_requirements(crop, soil_n, soil_p, soil_k, total_acres):
    """Calculate fertilizer requirements based on crop and soil NPK"""
    crop_req = crop_npk.get(crop, {"N": 100, "P": 50, "K": 50, "yield_per_acre": 1000, "price_per_kg": 20})
    
    # Convert hectare requirements to per acre
    crop_n_per_acre = crop_req["N"] / 2.471
    crop_p_per_acre = crop_req["P"] / 2.471
    crop_k_per_acre = crop_req["K"] / 2.471
    
    # Calculate soil NPK per acre
    soil_n_per_acre = soil_n / 2.471
    soil_p_per_acre = soil_p / 2.471
    soil_k_per_acre = soil_k / 2.471
    
    # Calculate deficits
    n_deficit = max(0, crop_n_per_acre - soil_n_per_acre * 0.3)
    p_deficit = max(0, crop_p_per_acre - soil_p_per_acre * 0.2)
    k_deficit = max(0, crop_k_per_acre - soil_k_per_acre * 0.4)
    
    # Calculate fertilizer amounts
    dap_for_p = p_deficit / 0.46  # DAP has 46% P2O5
    n_from_dap = dap_for_p * 0.18  # DAP has 18% N
    
    remaining_n = max(0, n_deficit - n_from_dap)
    urea_needed = remaining_n / 0.46  # Urea has 46% N
    
    mop_needed = k_deficit / 0.60  # MOP has 60% K2O
    
    # Optional: Potassium Nitrate for premium crops
    pot_nitrate_needed = 0
    if crop in ["Tomato", "Chilli", "Turmeric", "Onion"]:
        pot_nitrate_needed = k_deficit * 0.3 / 0.44
        mop_needed *= 0.7
    
    # Calculate totals
    total_urea = urea_needed * total_acres
    total_dap = dap_for_p * total_acres
    total_mop = mop_needed * total_acres
    total_pot_nitrate = pot_nitrate_needed * total_acres
    
    return {
        "urea_per_acre": round(urea_needed, 1),
        "dap_per_acre": round(dap_for_p, 1),
        "mop_per_acre": round(mop_needed, 1),
        "potassium_nitrate_per_acre": round(pot_nitrate_needed, 1),
        "total_urea": round(total_urea, 1),
        "total_dap": round(total_dap, 1),
        "total_mop": round(total_mop, 1),
        "crop_n": crop_req["N"],
        "crop_p": crop_req["P"],
        "crop_k": crop_req["K"]
    }

def calculate_water_requirements(crop, total_acres):
    """Calculate water requirements for the crop"""
    water_mm = water_requirements.get(crop, 500)
    water_per_acre_liters = water_mm * 10000  # Convert mm to liters/ha, then to per acre
    total_water = water_per_acre_liters * total_acres
    
    return {
        "water_per_acre": round(water_per_acre_liters),
        "total_water": round(total_water),
        "water_note": "Water requirement varies based on rainfall and irrigation method / వర్షపాతం మరియు నీటిపారుదల పద్ధతిని బట్టి నీటి అవసరం మారుతుంది"
    }

def predict_earnings_ml(district, season, soil, crop, soil_n, soil_p, soil_k, total_acres):
    """Predict earnings using ML model"""
    if model is None or any(enc is None for enc in encoders.values()):
        # Fallback calculation if model not loaded
        crop_req = crop_npk.get(crop, {"N": 100, "P": 50, "K": 50, "yield_per_acre": 1000, "price_per_kg": 20})
        soil_fertility = min(1.5, max(0.5, (soil_n/280 + soil_p/25 + soil_k/280)/3 * 1.5))
        expected_yield = crop_req["yield_per_acre"] * total_acres * soil_fertility
        return expected_yield * crop_req["price_per_kg"]
    
    try:
        # Encode inputs
        d_enc = encoders['district'].transform([district])[0]
        s_enc = encoders['season'].transform([season])[0]
        soil_enc = encoders['soil'].transform([soil])[0]
        crop_enc = encoders['crop'].transform([crop])[0]
        
        # Make prediction
        prediction = model.predict([[d_enc, s_enc, soil_enc, crop_enc, soil_n, soil_p, soil_k, total_acres]])[0]
        return prediction
    except Exception as e:
        print(f"ML prediction error: {e}")
        # Fallback calculation
        crop_req = crop_npk.get(crop, {"N": 100, "P": 50, "K": 50, "yield_per_acre": 1000, "price_per_kg": 20})
        soil_fertility = min(1.5, max(0.5, (soil_n/280 + soil_p/25 + soil_k/280)/3 * 1.5))
        expected_yield = crop_req["yield_per_acre"] * total_acres * soil_fertility
        return expected_yield * crop_req["price_per_kg"]

def get_alternative_crops(district, season, soil, soil_n, soil_p, soil_k, total_acres, current_crop):
    """Get top alternative crop predictions"""
    alternative_crops = []
    
    if model is not None and all(enc is not None for enc in encoders.values()):
        try:
            # Encode common inputs
            d_enc = encoders['district'].transform([district])[0]
            s_enc = encoders['season'].transform([season])[0]
            soil_enc = encoders['soil'].transform([soil])[0]
            
            for crop in crops:
                if crop != current_crop:
                    crop_enc = encoders['crop'].transform([crop])[0]
                    prediction = model.predict([[d_enc, s_enc, soil_enc, crop_enc, soil_n, soil_p, soil_k, total_acres]])[0]
                    alternative_crops.append((crop, prediction))
            
            # Sort by earnings and take top 3
            alternative_crops.sort(key=lambda x: x[1], reverse=True)
            return alternative_crops[:3]
        except Exception as e:
            print(f"Alternative crops prediction error: {e}")
    
    # Fallback: return some hardcoded alternatives
    return [
        ("Maize", 50000),
        ("Cotton", 45000),
        ("Groundnut", 40000)
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    
    if request.method == "POST":
        try:
            # Get form data
            district = request.form.get("district")
            season = request.form.get("season")
            soil = request.form.get("soil")
            soil_n = float(request.form.get("soil_n", 0))
            soil_p = float(request.form.get("soil_p", 0))
            soil_k = float(request.form.get("soil_k", 0))
            acres = float(request.form.get("acres", 0))
            guntas = float(request.form.get("guntas", 0))
            crop = request.form.get("crop")
            
            # Calculate total land area
            total_acres = acres + (guntas/40)
            
            # Validate inputs
            if total_acres <= 0:
                raise ValueError("Land area must be greater than 0")
            
            # Predict earnings using ML
            ml_earnings = predict_earnings_ml(district, season, soil, crop, soil_n, soil_p, soil_k, total_acres)
            
            # Calculate fertilizer requirements
            fertilizer_data = calculate_fertilizer_requirements(crop, soil_n, soil_p, soil_k, total_acres)
            
            # Calculate water requirements
            water_data = calculate_water_requirements(crop, total_acres)
            
            # Get alternative crops
            alternative_crops = get_alternative_crops(district, season, soil, soil_n, soil_p, soil_k, total_acres, crop)
            
            # Prepare result
            result = {
                "earnings": ml_earnings,
                "acres": acres,
                "guntas": guntas,
                "total_acres": total_acres,
                **fertilizer_data,
                **water_data,
                "top_crops": alternative_crops
            }
            
        except Exception as e:
            print(f"Error: {e}")
            result = None
    
    return render_template(
        "index.html",
        districts=districts,
        seasons=seasons,
        soils=soils,
        crops=crops,
        districts_te=districts_te,
        seasons_te=seasons_te,
        soils_te=soils_te,
        crops_te=crops_te,
        result=result
    )

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static/images", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("Starting Crop Yield Prediction Application...")
    print("Visit http://localhost:5000 in your browser")
    
app.run(host='0.0.0.0', port=5000)