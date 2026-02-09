import torch
import timm
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType, QuantFormat
import logging
from pathlib import Path

# --- Configuration ---
PYTORCH_MODEL_FILES = [f'model_fold_{i}.pth' for i in range(1, 6)]
MODEL_NAME = 'tf_efficientnet_b0'
NUM_CLASSES = 5 
MODEL_DIR = Path(__file__).parent

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_and_quantize_model(pytorch_model_path: Path):
    """
    Loads a PyTorch model, converts it to ONNX, and then quantizes it to INT8
    using a more compatible operator-oriented approach.
    """
    if not pytorch_model_path.is_file():
        logging.warning(f"Model file not found: {pytorch_model_path}. Skipping.")
        return

    logging.info(f"--- Starting conversion for {pytorch_model_path.name} ---")

    onnx_fp32_path = pytorch_model_path.with_suffix('.onnx')
    onnx_int8_path = pytorch_model_path.with_name(f"{pytorch_model_path.stem}_quant.onnx")

    # 1. Load PyTorch model
    logging.info("Step 1: Loading PyTorch model...")
    try:
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
        model.eval()
        logging.info("PyTorch model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load PyTorch model: {e}", exc_info=True)
        return

    # 2. Export to ONNX (FP32) with a newer opset
    logging.info(f"Step 2: Exporting to ONNX (FP32) -> {onnx_fp32_path.name}")
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_fp32_path),
            export_params=True,
            opset_version=14, 
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logging.info("ONNX FP32 export successful.")
    except Exception as e:
        logging.error(f"Failed to export to ONNX: {e}", exc_info=True)
        return

    # 3. Quantize the ONNX model (FP32 -> INT8)
    logging.info(f"Step 3: Quantizing to ONNX (INT8) -> {onnx_int8_path.name}")
    try:
        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            # --- THIS IS THE ONLY CHANGE ---
            # Using the parameter name your library version expects.
            op_types_to_quantize=['MatMul'],
            weight_type=QuantType.QInt8
        )
        logging.info("ONNX INT8 quantization successful.")
    except Exception as e:
        logging.error(f"Failed to quantize ONNX model: {e}", exc_info=True)
        return
    
    # 4. Clean up
    onnx_fp32_path.unlink()
    logging.info(f"Cleaned up intermediate file: {onnx_fp32_path.name}")
    logging.info(f"--- Finished conversion for {pytorch_model_path.name} ---")

if __name__ == "__main__":
    logging.info("Starting model conversion process for all specified models (Compatibility Mode).")
    for model_file in PYTORCH_MODEL_FILES:
        convert_and_quantize_model(MODEL_DIR / model_file)
    logging.info("All models processed.")
