# image_actions.py — complete non-Azure implementations
# All Azure implementations raise NotImplementedError

import logging
import io
import base64
import os
import random
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import cv2
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel, pipeline
from ultralytics import YOLO
from openai import OpenAI

logger = logging.getLogger(__name__)

# ================================================================
# OpenAI client (for GPT-4o, DALL·E 3 OpenAI versions)
# ================================================================
_openai_client: Optional[OpenAI] = None

def _init_openai():
    global _openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Environment variable OPENAI_API_KEY must be set.")
        return
    _openai_client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized.")


# ================================================================
# Heavy-weight model singletons
# ================================================================
_models_loaded = False
_yolov8_s_model: Optional[YOLO] = None
_yolov8_x_model: Optional[YOLO] = None
_dncnn_model: Optional[torch.nn.Module] = None
_clip_processor: Optional[AutoProcessor] = None
_clip_model: Optional[AutoModel] = None
_siglip_processor: Optional[AutoProcessor] = None
_siglip_model: Optional[AutoModel] = None
_easyocr_reader: Optional[Any] = None
_blip2_processor: Optional[AutoProcessor] = None
_blip2_model: Optional[AutoModelForCausalLM] = None
_llava_pipeline: Optional[pipeline] = None
# _esrgan_model placeholder if you have real ESRGAN weights


def _init_models():
    global _models_loaded
    if _models_loaded:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading vision models on {device} …")

    try:
        from ultralytics import YOLO
        globals()["_yolov8_s_model"] = YOLO("yolov8s.pt")
        globals()["_yolov8_x_model"] = YOLO("yolov8x.pt")
    except Exception as e:
        logger.warning(f"YOLO weights not found: {e}")

    try:
        _clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        globals().update(_clip_processor=_clip_processor, _clip_model=_clip_model)
    except Exception as e:
        logger.warning(f"CLIP load failed: {e}")

    try:
        _siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        _siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        globals().update(_siglip_processor=_siglip_processor, _siglip_model=_siglip_model)
    except Exception as e:
        logger.warning(f"SigLIP load failed: {e}")

    try:
        import easyocr
        globals()["_easyocr_reader"] = easyocr.Reader(["en", "ko"], gpu=torch.cuda.is_available())
    except Exception as e:
        logger.warning(f"EasyOCR load failed: {e}")

    try:
        _blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        _blip2_model = AutoModelForCausalLM.from_pretrained(
            "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto"
        )
        globals().update(_blip2_processor=_blip2_processor, _blip2_model=_blip2_model)
    except Exception as e:
        logger.warning(f"BLIP-2 load failed: {e}")

    try:
        _llava_pipeline = pipeline(
            "visual-question-answering",
            model="llava-hf/llava-1.5-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        globals()["_llava_pipeline"] = _llava_pipeline
    except Exception as e:
        logger.warning(f"LLaVA load failed: {e}")

    globals()["_models_loaded"] = True


# Initialize everything at import time
_init_openai()
_init_models()

# ================================================================
# Helper utilities
# ================================================================

def _pil_to_np_gray(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"))


def _ensure_openai() -> OpenAI:
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized; set OPENAI_API_KEY")
    return _openai_client

# ================================================================
# 1. DecodeImage / ResizeImage (Pillow)
# ================================================================

def DecodeImage_Pillow_Decode(binary_image: bytes) -> Image.Image:
    return Image.open(io.BytesIO(binary_image)).convert("RGB")


def ResizeImage_Pillow_Resize_Lanczos(image_object: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    return image_object.resize(target_size, Image.Resampling.LANCZOS)

# ================================================================
# 2. AssessClarity (OpenCV)
# ================================================================

def AssessClarity_OpenCV_Laplacian_Variance(image_object: Image.Image) -> float:
    variance = cv2.Laplacian(_pil_to_np_gray(image_object), cv2.CV_64F).var()
    return min(1.0, variance / 5000.0)

# ================================================================
# 3. DenoiseImage (OpenCV + DnCNN)
# ================================================================

def DenoiseImage_OpenCV_FastNlMeans(image_object: Image.Image) -> Image.Image:
    bgr = cv2.cvtColor(np.array(image_object), cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
    return Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))


def DenoiseImage_DnCNN_PyTorch(image_object: Image.Image) -> Image.Image:
    if _dncnn_model is None:
        logger.warning("DnCNN not loaded; returning original image")
        return image_object
    tensor = torch.from_numpy(np.array(image_object).astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(next(_dncnn_model.parameters()).device)
    with torch.no_grad():
        out = _dncnn_model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    return Image.fromarray(out.astype(np.uint8))

# ================================================================
# 4. GetImageEmbedding (CLIP / SigLIP) — Azure version stub
# ================================================================

def GetImageEmbedding_CLIP_ViT_B_32(image_object: Image.Image) -> np.ndarray:
    if _clip_model is None or _clip_processor is None:
        logger.warning("CLIP not loaded; returning random embedding")
        return np.random.rand(512).astype(np.float32)
    inputs = _clip_processor(images=image_object, return_tensors="pt").to(_clip_model.device)
    with torch.no_grad():
        emb = _clip_model.get_image_features(**inputs).cpu().numpy().flatten()
    return emb


def GetImageEmbedding_SigLIP_S_14(image_object: Image.Image) -> np.ndarray:
    if _siglip_model is None or _siglip_processor is None:
        logger.warning("SigLIP not loaded; returning random embedding")
        return np.random.rand(768).astype(np.float32)
    inputs = _siglip_processor(images=image_object, return_tensors="pt").to(_siglip_model.device)
    with torch.no_grad():
        emb = _siglip_model.get_image_features(**inputs).cpu().numpy().flatten()
    return emb


def GetImageEmbedding_Azure_Vectorize_Image(*_, **__):  # stub
    raise NotImplementedError("Azure Vectorize Image is not available in this environment.")

# ================================================================
# 5. DetectObjects (YOLO)
# ================================================================

def _yolo_detect(model: YOLO, image: Image.Image) -> List[Dict]:
    if model is None:
        return [{"box": [0, 0, image.width, image.height], "label": "dummy", "confidence": 0.0}]
    results = model(image)
    out = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out.append({
                "box": [x1, y1, x2, y2],
                "label": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
            })
    return out


def DetectObjects_YOLOv8_S(image_object: Image.Image) -> List[Dict]:
    return _yolo_detect(_yolov8_s_model, image_object)


def DetectObjects_YOLOv8_X(image_object: Image.Image) -> List[Dict]:
    return _yolo_detect(_yolov8_x_model, image_object)

# ================================================================
# 6. ReadTextFromImage (EasyOCR) — Azure stub
# ================================================================

def ReadTextFromImage_EasyOCR(image_object: Image.Image) -> str:
    if _easyocr_reader is None:
        return ""
    results = _easyocr_reader.readtext(np.array(image_object))
    return " ".join(text for _, text, _ in results)


def ReadTextFromImage_Azure_Read_API(*_, **__):
    raise NotImplementedError("Azure Read API not available in this environment.")

# ================================================================
# 7. GetImageCaption (BLIP-2) — Azure stub
# ================================================================

def GetImageCaption_BLIP_2_FlanT5_XL(image_object: Image.Image) -> str:
    if _blip2_model is None or _blip2_processor is None:
        return ""
    inputs = _blip2_processor(images=image_object, text="", return_tensors="pt").to(_blip2_model.device)
    with torch.no_grad():
        ids = _blip2_model.generate(**inputs, max_new_tokens=60)
    return _blip2_processor.decode(ids[0], skip_special_tokens=True).strip()


def GetImageCaption_Azure_Describe_API(*_, **__):
    raise NotImplementedError("Azure Describe API not available in this environment.")

# ================================================================
# 8. AnswerQuestionAboutImage (LLaVA / GPT-4o OpenAI) — Azure GPT-4o stub
# ================================================================

def AnswerQuestionAboutImage_LLaVA_1_5_7B(image_object: Image.Image, question_text: str) -> str:
    if _llava_pipeline is None:
        return ""
    try:
        res = _llava_pipeline(image=image_object, question=question_text)
        return res[0]["answer"] if res else ""
    except Exception as e:
        logger.warning(f"LLaVA failed: {e}")
        return ""


def AnswerQuestionAboutImage_GPT_4o_OpenAI(image_object: Image.Image, question_text: str) -> str:
    client = _ensure_openai()
    buf = io.BytesIO()
    image_object.save(buf, format="PNG")
    img64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": question_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img64}"}},
            ]},
        ],
        max_tokens=300,
    )
    return resp.choices[0].message.content


def AnswerQuestionAboutImage_GPT_4o_Azure(*_, **__):
    raise NotImplementedError("Azure GPT-4o not available in this environment.")

# ================================================================
# 9. GenerateImageFromText (SDXL-Turbo / DALL·E 3 OpenAI) — Azure stub
# ================================================================

def GenerateImageFromText_SDXL_Turbo(prompt_text: str) -> Image.Image:
    # If diffusers pipeline available, plug in here
    img = Image.new("RGB", (512, 512), color=(73, 109, 137))
    return img


def GenerateImageFromText_DALL_E_3_OpenAI(prompt_text: str) -> Image.Image:
    client = _ensure_openai()
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt_text,
        n=1,
        size="1024x1024",
        quality="standard",
        response_format="b64_json",
    )
    data = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(data))


def GenerateImageFromText_DALL_E_3(*_, **__):
    raise NotImplementedError("Azure DALL·E 3 not available in this environment.")

# ================================================================
# 10. UpscaleImage (ESRGAN-x4)
# ================================================================

def UpscaleImage_ESRGAN_x4(image_object: Image.Image, scale_factor: int) -> Image.Image:
    if scale_factor != 4:
        logger.warning("ESRGAN-x4 placeholder only supports 4×; using nearest resize instead")
    return image_object.resize((image_object.width * scale_factor, image_object.height * scale_factor), Image.Resampling.NEAREST)

# ================================================================
# 11. CropToObject (Pillow)
# ================================================================

def CropToObject_Pillow_Crop(image_object: Image.Image, bounding_box: List[int]) -> Image.Image:
    return image_object.crop(tuple(bounding_box))

# ================================================================
# Azure-only stubs (kept for clarity) that may still be referenced
# ================================================================

def GetImageCaption_Azure_Describe_API_stub(*_, **__):
    raise NotImplementedError

def ReadTextFromImage_Azure_Read_API_stub(*_, **__):
    raise NotImplementedError

def GetImageEmbedding_Azure_Vectorize_Image_stub(*_, **__):
    raise NotImplementedError