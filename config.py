from pathlib import Path

# Base Path: works seamlessly both inside Docker (/home/TypeCare) and in any cloned directory
ours = Path(__file__).resolve().parent

# Prediction Path
PRED_PATH = ours / "prediction"

# Data Path
DATA_PATH = ours / "data"

# Output Path
OUTPUT_PATH = ours / "output"

# Model Path
MODEL_PATH = PRED_PATH / "typesim_model"


TIGER_TRANSFORM_PATH = PRED_PATH / "Tiger" / "transformed_result.json"
TIGER_OUTPUT_PATH = OUTPUT_PATH / "tiger_result.pkl"


# Configuration for TypeGen
TYPEGEN_PATH= PRED_PATH / "TypeGen"
TYPEGEN_PREDICTION_PATH = TYPEGEN_PATH / "data" / "predictions" / "typegen.json"
TYPEGEN_TRANSFORM_PATH = PRED_PATH / "TypeGen" / "transformed_result.json"
TYPEGEN_OUTPUT_PATH = OUTPUT_PATH / "typegen_result.pkl"

# Configuration for TypeT5
TYPET5_PATH = PRED_PATH / "TypeT5"
TYPET5_PREDICTION_PATH = PRED_PATH / "TypeT5" / 'double-traversal-EvalResultAllTest_0329.pkl'
TYPET5_TRANSFORM_PATH = PRED_PATH / "TypeT5" / "transformed_result.json"
TYPET5_OUTPUT_PATH = OUTPUT_PATH / "typet5_result.pkl"

# Configuration for Example
EXAMPLE_TRANSFORM_PATH = PRED_PATH / "Example" / "transformed_result.json"
EXAMPLE_OUTPUT_PATH = OUTPUT_PATH / "example_result.pkl"

# Configuration for Type4Py
TYPE4PY_PATH = PRED_PATH / "Type4Py"
TYPE4PY_TRANSFORM_PATH = PRED_PATH / "Type4Py" / "transformed_result.json"
TYPE4PY_OUTPUT_PATH = OUTPUT_PATH / "type4py_result.pkl"
