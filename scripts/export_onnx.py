# src/credit_scoring/serving/export_onnx.py
"""
Exporte le modèle LightGBM chargé via MLflow en format ONNX.
Usage : python -m credit_scoring.serving.export_onnx
"""

from pathlib import Path

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

from credit_scoring.config import PROD_MODEL
from credit_scoring.serving.inference import get_model


def export():
    model, features, threshold = get_model()

    n_features = len(features)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
    )

    output_path = Path(PROD_MODEL) / "model.onnx"
    onnxmltools.utils.save_model(onnx_model, str(output_path))
    print(f"✅ Modèle ONNX exporté → {output_path}")
    print(f"   Features : {n_features}")
    print(f"   Threshold : {threshold}")


if __name__ == "__main__":
    export()
