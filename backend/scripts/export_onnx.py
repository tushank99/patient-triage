import os
import joblib
import onnx
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "xgboost_triage_model.joblib")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "frontend", "public")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "triage.onnx")

print(f"Loading trained XGBoost model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Cannot find model at {MODEL_PATH}. Run your train_triage_model.py script first!")

# Load the joblib model
xgb_model = joblib.load(MODEL_PATH)

#  Define the exact 8-feature tensor input required by ONNX:
#  [age, hr, bp_sys, bp_dia, spo2, temp, pain, has_history]
initial_type = [('float_input', FloatTensorType([None, 8]))]

print("Converting XGBoost model to ONNX...")
onnx_model = convert_xgboost(xgb_model, initial_types=initial_type)

#  Save directly to the frontend's public folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx.save_model(onnx_model, OUTPUT_FILE)

print(f" Successfully exported ONNX model to: {OUTPUT_FILE}")