import asyncio
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision import transforms
from app.utils.logger import logger
import config


class ModelService:
    def __init__(self):
        self.disease_models = {}
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.class_names = {
            0: "Coccidiosis",
            1: "Botulism",
            2: "Newcastle Disease",
            3: "Chickenpox",
            4: "Lice and Mites",
        }

    async def initialize(self, executor):
        loop = asyncio.get_running_loop()
        paths = [
            config.STATIC_DIR / f"diseases/model_fold_{i}_quant.onnx"
            for i in range(1, 6)
        ]
        load_tasks = []
        for p in paths:
            if p.is_file():
                load_tasks.append(
                    loop.run_in_executor(
                        executor,
                        lambda path=p: (
                            path.name,
                            ort.InferenceSession(
                                str(path), providers=["CPUExecutionProvider"]
                            ),
                        ),
                    )
                )
            else:
                logger.warning(f"ONNX model not found: {p}")

        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to load ONNX model: {result}")
                continue
            name, session = result
            self.disease_models[name] = session

        logger.info(f"Loaded {len(self.disease_models)} disease models.")

    def classify(self, crop):
        if not self.disease_models:
            return None, 0.0
        if crop is None or crop.size == 0:
            return None, 0.0
        try:
            if len(crop.shape) != 3 or crop.shape[2] != 3:
                logger.debug(f"Unexpected crop shape: {crop.shape}")
                return None, 0.0
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = self.transforms(img).unsqueeze(0).numpy()
            best_conf, best_name = -1.0, "Low Confidence"
            for _, session in self.disease_models.items():
                out = session.run(None, {"input": tensor})[0]
                # Numerically stable softmax
                out_shifted = out - np.max(out, axis=1, keepdims=True)
                exp_scores = np.exp(out_shifted)
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                conf = float(np.max(probs))
                pred_idx = int(np.argmax(probs))
                if conf > best_conf:
                    best_conf = conf
                    best_name = self.class_names.get(pred_idx, "Unknown")
            if best_conf < config.CLASSIFICATION_CONFIDENCE_THRESHOLD:
                return "Low Confidence", best_conf
            return best_name, best_conf
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return None, 0.0
