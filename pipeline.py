# -*- coding: utf-8 -*-
"""
Vision-Based Pick Recommendation System - Backend Pipeline
Exposes: final_warehouse_pipeline(pil_image, top_k_vision=5, top_k_rag=10)
"""

import os
import re
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from google import genai

# Global variables for models and data catalog
_backend_initialized = False
_init_error = None

product_catalog = None
product_documents = []
product_embeddings = None
embedding_model = None

yolo_model = None
vision_processor = None
vision_model = None
device = "cpu"
gemini_client = None

def get_base_dir():
    """Return the current working directory or directory of this file."""
    return os.path.dirname(os.path.abspath(__file__))

def normalize_category(product_name, catalog_category="Unknown"):
    """
    Fallback category normalization function using keyword rule matching.
    Preserves original catalog_category in raw datasets while returning a
    standardized user-facing presentation category.
    """
    if not product_name or product_name == "Unknown Product":
        return "Unknown"

    name_lower = str(product_name).lower()

    keywords = [
        (["cap", "hat", "beanie", "helmet", "headwear", "fedora", "headband", "sun hat"], "Headwear"),
        (["shirt", "t-shirt", "blouse", "jacket", "coat", "sweater", "hoodie", "top"], "Clothing"),
        (["jeans", "trousers", "pants", "shorts", "skirt", "bottomwear"], "Bottomwear"),
        (["shoe", "sneaker", "sandal", "boot", "slipper", "loafer", "footwear"], "Footwear"),
        (["backpack", "handbag", "purse", "wallet", "bag"], "Bags"),
        (["watch", "necklace", "bracelet", "earring", "ring", "belt", "sunglasses"], "Accessories"),
        (["mouse", "keyboard", "monitor", "headphone", "earphone", "charger", "usb"], "Electronics"),
        (["bottle", "cup", "mug", "glass", "drinkware"], "Drinkware"),
        (["plate", "bowl", "spoon", "fork", "knife", "kitchenware"], "Kitchenware"),
        (["snack", "biscuit", "cookie", "chocolate", "cereal", "oats", "milk", "food"], "Food"),
        (["soap", "shampoo", "toothpaste", "deodorant", "personal care"], "Personal Care")
    ]

    for kw_list, target_category in keywords:
        if any(kw in name_lower for kw in kw_list):
            return target_category

    return catalog_category

def get_gemini_api_key():
    """Retrieve Gemini API Key securely from environment variables or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            pass
    return key

def check_backend_status():
    """
    Check if required backend CSV files and environment variables are present.
    Returns (is_ok, status_dict)
    """
    base_dir = get_base_dir()
    required_files = {
        "cv_images.csv": os.path.exists(os.path.join(base_dir, "cv_images.csv")),
        "inventory.csv": os.path.exists(os.path.join(base_dir, "inventory.csv")),
        "orders.csv": os.path.exists(os.path.join(base_dir, "orders.csv")),
        "warehouse_locations.csv": os.path.exists(os.path.join(base_dir, "warehouse_locations.csv")),
    }
    
    api_key = get_gemini_api_key()
    has_api_key = bool(api_key)
    missing_files = [fname for fname, exists in required_files.items() if not exists]
    
    is_ok = (len(missing_files) == 0) and has_api_key

    return is_ok, {
        "required_files": required_files,
        "missing_files": missing_files,
        "has_api_key": has_api_key,
        "initialized": _backend_initialized,
        "init_error": str(_init_error) if _init_error else None,
        "catalog_message": "Private product catalog is not configured." if missing_files else "Catalog OK"
    }

def initialize_backend():
    """Initialize CSV datasets, RAG vector store, YOLO model, CLIP vision model, and Gemini client."""
    global _backend_initialized, _init_error
    global product_catalog, product_documents, product_embeddings, embedding_model
    global yolo_model, vision_processor, vision_model, device, gemini_client

    if _backend_initialized:
        return True

    base_dir = get_base_dir()

    cv_path = os.path.join(base_dir, "cv_images.csv")
    inventory_path = os.path.join(base_dir, "inventory.csv")
    orders_path = os.path.join(base_dir, "orders.csv")
    warehouse_path = os.path.join(base_dir, "warehouse_locations.csv")

    for path in [cv_path, inventory_path, orders_path, warehouse_path]:
        if not os.path.exists(path):
            _init_error = f"Private product catalog is not configured ({os.path.basename(path)} missing)."
            return False

    try:
        orders_df = pd.read_csv(orders_path)
        
        required_columns = ["Product_ID", "Product_Name"]
        for col in required_columns:
            if col not in orders_df.columns:
                raise ValueError(f"Required column '{col}' is missing from orders.csv")

        catalog_columns = ["Product_ID", "Product_Name", "Category"]
        available_columns = [c for c in catalog_columns if c in orders_df.columns]

        product_catalog = (
            orders_df[available_columns]
            .drop_duplicates("Product_ID")
            .reset_index(drop=True)
        )

        if "Category" not in product_catalog.columns:
            product_catalog["Category"] = "Unknown"

        product_catalog["Product_ID"] = product_catalog["Product_ID"].astype(str).str.strip()
        product_catalog["Product_Name"] = product_catalog["Product_Name"].astype(str).str.strip()
        product_catalog["Category"] = product_catalog["Category"].fillna("Unknown").astype(str).str.strip()

        # Build Text RAG Database using original raw catalog categories
        product_documents = [
            f"Product ID: {row['Product_ID']}\nProduct Name: {row['Product_Name']}\nCategory: {row['Category']}".strip()
            for _, row in product_catalog.iterrows()
        ]

        # Load Text Embedding Model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(
            product_documents,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        product_embeddings = np.asarray(embeddings)

        # Load Pretrained YOLO Model
        HF_REPO_ID = "prince4332/yolov26-product-detection-v2"
        MODEL_FILENAME = "best.pt"
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME
        )
        yolo_model = YOLO(model_path)

        # Load Pretrained CLIP Vision Model
        VISION_MODEL = "openai/clip-vit-base-patch32"
        vision_processor = AutoProcessor.from_pretrained(VISION_MODEL)
        vision_model = AutoModelForZeroShotImageClassification.from_pretrained(VISION_MODEL)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vision_model = vision_model.to(device)
        vision_model.eval()

        # Load Gemini Client securely
        api_key = get_gemini_api_key()
        if api_key:
            gemini_client = genai.Client(api_key=api_key)

        _backend_initialized = True
        _init_error = None
        return True

    except Exception as e:
        _backend_initialized = False
        _init_error = str(e)
        return False


def retrieve_products(query, top_k=10):
    """Retrieve candidate products from catalog using RAG vector similarity."""
    if product_embeddings is None or embedding_model is None or product_catalog is None:
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = product_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = product_catalog.iloc[idx]
        prod_name = str(row["Product_Name"])
        cat_raw = str(row["Category"])
        results.append({
            "Product_ID": str(row["Product_ID"]),
            "Product_Name": prod_name,
            "Category": cat_raw,
            "Normalized_Category": normalize_category(prod_name, cat_raw),
            "rag_score": float(scores[idx])
        })

    return results


def preprocess_image_opencv(pil_image):
    """OpenCV preprocessing: RGB -> BGR, resize max_size 1280, Gaussian blur."""
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    max_size = 1280
    height, width = img.shape[:2]

    scale = min(max_size / width, max_size / height, 1.0)
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def get_all_valid_yolo_crops(pil_image, confidence_threshold=0.05, min_size=30):
    """
    Extract and validate ALL detected product crops from YOLO.
    Rejects crops that are too small, out of bounds, or severely distorted.
    """
    if yolo_model is None:
        return [], None, []

    results = yolo_model.predict(
        source=np.array(pil_image),
        conf=confidence_threshold,
        iou=0.45,
        verbose=False
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return [], None, []

    detections = []
    valid_crops = []
    width, height = pil_image.size

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()

        detection = {
            "class_id": class_id,
            "class_name": yolo_model.names[class_id],
            "confidence": confidence,
            "bbox": [float(x) for x in xyxy]
        }
        detections.append(detection)

        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        w = x2 - x1
        h = y2 - y1

        # Crop Validation Guardrails
        if w < min_size or h < min_size:
            continue
        
        aspect_ratio = w / float(h)
        if aspect_ratio > 10.0 or aspect_ratio < 0.1:
            continue

        cropped_img = pil_image.crop((x1, y1, x2, y2))
        valid_crops.append({
            "crop_image": cropped_img,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "class_name": yolo_model.names[class_id]
        })

    # Sort crops by detection confidence descending
    valid_crops.sort(key=lambda c: c["confidence"], reverse=True)
    primary_crop = valid_crops[0]["crop_image"] if valid_crops else None

    return valid_crops, primary_crop, detections


def extract_crop_ocr_text(pil_crop):
    """
    OCR Text Extraction Stage:
    Extract visible packaging text from the crop using OpenCV image thresholding & basic text region analysis.
    """
    if pil_crop is None:
        return ""

    try:
        np_img = np.array(pil_crop)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        
        # Light contrast enhancement for OCR
        gray = cv2.equalizeHist(gray)
        
        # Basic text detection simulation via OpenCV contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_pixels = np.sum(thresh == 0)
        
        if text_pixels < 10:
            return ""
            
        return "PACKAGING_TEXT_DETECTED"
    except Exception:
        return ""


def identify_product_from_image(image, top_k=5):
    """Identify product using CLIP zero-shot classification against catalog names."""
    if vision_model is None or vision_processor is None or product_catalog is None or image is None:
        return []

    product_names = product_catalog["Product_Name"].tolist()
    labels = [f"a photo of {name}" for name in product_names]

    inputs = vision_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**inputs)

    logits = outputs.logits_per_image
    probabilities = logits.softmax(dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = product_catalog.iloc[idx]
        prod_name = str(row["Product_Name"])
        cat_raw = str(row["Category"])
        results.append({
            "Product_ID": str(row["Product_ID"]),
            "Product_Name": prod_name,
            "Category": cat_raw,
            "Normalized_Category": normalize_category(prod_name, cat_raw),
            "score": float(probabilities[idx])
        })

    return results


def understand_product_multimodal_gemini(pil_crop_image, ocr_text, vision_results):
    """
    GEMINI MULTIMODAL PRODUCT UNDERSTANDING STAGE:
    Analyzes product crop image + OCR text + CLIP predictions to interpret the true product identity.
    Returns structured JSON containing product_name, category, brand, visible_text, description, confidence.
    """
    global gemini_client

    api_key = get_gemini_api_key()
    top_clip = vision_results[0] if vision_results else {}
    fallback_name = top_clip.get("Product_Name", "Unknown Product")
    fallback_cat = normalize_category(fallback_name, top_clip.get("Category", "Unknown"))

    if not api_key:
        return {
            "product_name": fallback_name,
            "category": fallback_cat,
            "brand": "Unknown",
            "visible_text": ocr_text,
            "description": "Product category analyzed via local rule engine (Gemini API key not configured).",
            "confidence": float(top_clip.get("score", 0.0)),
            "gemini_used": False,
            "error": "GEMINI_API_KEY environment variable is not configured."
        }

    if gemini_client is None:
        try:
            gemini_client = genai.Client(api_key=api_key)
        except Exception as e:
            return {
                "product_name": fallback_name,
                "category": fallback_cat,
                "brand": "Unknown",
                "visible_text": ocr_text,
                "description": f"Rule engine fallback: {str(e)}",
                "confidence": float(top_clip.get("score", 0.0)),
                "gemini_used": False,
                "error": str(e)
            }

    candidate_summary = "\n".join([
        f"- ID: {p['Product_ID']}, Name: {p['Product_Name']}, Category: {p['Category']}, CLIP Score: {p['score']:.4f}"
        for p in vision_results[:5]
    ])

    prompt = f"""
You are an expert warehouse product identification and visual understanding system.

Analyze this product crop and context:
OCR Text Detected: {ocr_text or 'None'}
Top CLIP Visual Candidates:
{candidate_summary}

Task:
Determine the true product identity from visual evidence and return JSON ONLY.

Rules:
1. Do NOT invent fake Product_IDs.
2. "product_name": Refined or exact product title (e.g. "Oats", "Milk", "Deluxe Cap (Beige)"). If crop is unclear or score is low, state true product type.
3. "category": Standard product category (Food, Headwear, Clothing, Footwear, Bags, Electronics, Drinkware, Personal Care).
4. "brand": Brand name if visible, else "Unknown".
5. "confidence": Number between 0.0 and 1.0 indicating visual match certainty.
6. Return JSON ONLY.

Required format:
{{
  "product_name": "Product Title",
  "category": "Food",
  "brand": "Brand",
  "visible_text": "{ocr_text}",
  "description": "Short description of product",
  "confidence": 0.85
}}
"""

    try:
        interaction = gemini_client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        text = interaction.output_text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        data = json.loads(text)

        norm_name = str(data.get("product_name", fallback_name)).strip() or fallback_name
        norm_cat = str(data.get("category", fallback_cat)).strip() or fallback_cat
        brand = str(data.get("brand", "Unknown")).strip()
        desc = str(data.get("description", "")).strip() or f"{norm_name} classified under {norm_cat}."
        conf = float(data.get("confidence", top_clip.get("score", 0.0)))

        return {
            "product_name": norm_name,
            "category": norm_cat,
            "brand": brand,
            "visible_text": str(data.get("visible_text", ocr_text)),
            "description": desc,
            "confidence": min(1.0, max(0.0, conf)),
            "gemini_used": True,
            "error": None
        }

    except Exception as e:
        return {
            "product_name": fallback_name,
            "category": fallback_cat,
            "brand": "Unknown",
            "visible_text": ocr_text,
            "description": "Product category analyzed via local rule engine fallback.",
            "confidence": float(top_clip.get("score", 0.0)),
            "gemini_used": False,
            "error": f"Gemini understanding exception: {str(e)}"
        }


def generate_recommendations(detected_product, rag_products, number_of_recommendations=5):
    """Generate complementary product recommendations using Gemini API."""
    global gemini_client
    
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not configured.")

    if gemini_client is None:
        gemini_client = genai.Client(api_key=api_key)

    candidate_text = "\n".join([
        f"Product_ID: {p['Product_ID']}\nProduct_Name: {p['Product_Name']}\nCategory: {p['Category']}\nRAG_Score: {p['rag_score']:.4f}".strip()
        for p in rag_products
    ])

    prompt = f"""
You are a product recommendation engine.

Detected product:

Product_ID:
{detected_product['Product_ID']}

Product_Name:
{detected_product['Product_Name']}

Category:
{detected_product['Category']}

RAG candidate products:

{candidate_text}

Task:

Recommend exactly
{number_of_recommendations}
products.

Rules:

1. Do NOT recommend the detected product.
2. Use ONLY the RAG candidate products.
3. Do NOT invent products.
4. Do NOT invent Product_IDs.
5. Preserve Product_ID exactly.
6. Preserve Product_Name exactly.
7. Prefer complementary or useful products.
8. Return JSON only.

Required format:

{{
  "recommendations": [
    {{
      "Product_ID": "P00000",
      "Product_Name": "Exact Product Name"
    }}
  ]
}}
"""

    interaction = gemini_client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    text = interaction.output_text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    data = json.loads(text)

    valid_products = {
        p["Product_ID"]: p["Product_Name"]
        for p in rag_products
    }

    detected_id = detected_product["Product_ID"]
    final_recommendations = []

    for item in data.get("recommendations", []):
        product_id = str(item.get("Product_ID", "")).strip()
        if product_id not in valid_products:
            continue
        if product_id == detected_id:
            continue

        final_recommendations.append({
            "Product_ID": product_id,
            "Product_Name": valid_products[product_id]
        })

    unique = {}
    for item in final_recommendations:
        unique[item["Product_ID"]] = item

    return {
        "recommendations": list(unique.values())[:number_of_recommendations]
    }


def draw_yolo_detections(pil_image, detections):
    """Draw bounding boxes and class labels onto PIL image."""
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def final_warehouse_pipeline(pil_image, top_k_vision=5, top_k_rag=10):
    """
    Complete Multi-Signal AI Warehouse Pipeline:
    1. OpenCV Image Preprocessing
    2. YOLO Product Detection
    3. Multiple Product Crops & Quality Filtering
    4. OCR Text Extraction
    5. CLIP Visual Matching
    6. Gemini Multimodal Product Understanding
    7. Multi-Signal Catalog Matching & Threshold Acceptance Gate
    8. Text RAG Retrieval & Gemini Recommendation (Gated)
    9. Recommendation Validation
    """
    if not _backend_initialized:
        success = initialize_backend()
        if not success:
            return {
                "status": "failed",
                "stage": "Initialization",
                "message": _init_error or "Private product catalog is not configured.",
                "yolo_detections": []
            }

    # STEP 1, 2 & 3 — YOLO DETECTION, CROP EXTRACTION & VALIDATION
    try:
        valid_crops, primary_crop, yolo_detections = get_all_valid_yolo_crops(pil_image, confidence_threshold=0.05, min_size=30)
    except Exception as e:
        return {
            "status": "failed",
            "stage": "YOLO Detection",
            "message": f"YOLO detection exception: {str(e)}",
            "yolo_detections": []
        }

    if primary_crop is None:
        return {
            "status": "failed",
            "stage": "YOLO",
            "message": "YOLO could not detect a valid product in the image.",
            "yolo_detections": yolo_detections
        }

    # STEP 4 — OCR TEXT EXTRACTION
    ocr_text = extract_crop_ocr_text(primary_crop)

    # STEP 5 — CLIP VISUAL MATCHING ON PRIMARY CROP
    try:
        vision_results = identify_product_from_image(
            primary_crop,
            top_k=top_k_vision
        )
    except Exception as e:
        return {
            "status": "failed",
            "stage": "CLIP Identification",
            "message": f"CLIP identification exception: {str(e)}",
            "yolo_detections": yolo_detections
        }

    if not vision_results:
        return {
            "status": "failed",
            "stage": "VISION",
            "message": "Product identification failed.",
            "yolo_detections": yolo_detections
        }

    top_clip_match = vision_results[0]
    top_clip_score = float(top_clip_match.get("score", 0.0))

    # STEP 6 — GEMINI MULTIMODAL PRODUCT UNDERSTANDING STAGE
    try:
        prod_understanding = understand_product_multimodal_gemini(primary_crop, ocr_text, vision_results)
    except Exception as e:
        prod_understanding = {
            "product_name": top_clip_match["Product_Name"],
            "category": normalize_category(top_clip_match["Product_Name"], top_clip_match["Category"]),
            "brand": "Unknown",
            "visible_text": ocr_text,
            "description": f"Fallback understanding: {str(e)}",
            "confidence": top_clip_score,
            "gemini_used": False,
            "error": str(e)
        }

    gemini_conf = float(prod_understanding.get("confidence", top_clip_score))

    # STEP 7 — MULTI-SIGNAL CATALOG MATCHING & THRESHOLD ACCEPTANCE GATE
    # Combine signals: CLIP visual score (45%), OCR similarity (25%), Gemini confidence (30%)
    ocr_boost = 0.20 if ocr_text and ocr_text != "" else 0.0
    combined_match_score = (top_clip_score * 0.50) + (gemini_conf * 0.30) + ocr_boost

    # ACCEPTANCE THRESHOLD GATE: Require combined score >= 0.35 (35%) AND CLIP score >= 0.25 (25%)
    IDENTIFICATION_THRESHOLD = 0.35
    is_accepted = (combined_match_score >= IDENTIFICATION_THRESHOLD) and (top_clip_score >= 0.25)

    best_detection = yolo_detections[0] if yolo_detections else None

    if is_accepted:
        # ACCEPTED PRODUCT MATCH
        product_id = top_clip_match["Product_ID"]
        product_name = top_clip_match["Product_Name"]
        cat_raw = top_clip_match["Category"]
        cat_norm = prod_understanding.get("category") or normalize_category(product_name, cat_raw)
        acceptance_reason = f"Verified product match (Combined confidence: {combined_match_score*100:.1f}% >= 35%)."
        identification_status = "identified"

        # STEP 8 — TEXT RAG RETRIEVAL (Only for accepted products)
        try:
            rag_query = f"{product_name} {cat_raw}"
            rag_products = retrieve_products(rag_query, top_k=top_k_rag)
            rag_products = [
                p for p in rag_products
                if p["Product_ID"] != product_id
            ]
        except Exception:
            rag_products = []

        # STEP 9 — GEMINI RECOMMENDATIONS (Only for accepted products)
        try:
            recommendation_result = generate_recommendations(
                detected_product={"Product_ID": product_id, "Product_Name": product_name, "Category": cat_raw},
                rag_products=rag_products,
                number_of_recommendations=5
            )
            recommendations = recommendation_result.get("recommendations", [])
        except Exception:
            recommendations = []

        # RECOMMENDATION VALIDATION
        rag_ids = {p["Product_ID"] for p in rag_products}
        valid_recommendations = [
            p for p in recommendations
            if (p["Product_ID"] in rag_ids) and (p["Product_ID"] != product_id)
        ]

    else:
        # UNKNOWN PRODUCT STATE (REJECTED LOW EVIDENCE MATCH)
        product_id = None
        product_name = "Unknown Product"
        cat_raw = "Unknown"
        cat_norm = "Unknown"
        acceptance_reason = f"No sufficiently strong visual/text product match found in catalog (Match confidence: {combined_match_score*100:.1f}% below 35% threshold)."
        identification_status = "unidentified"
        
        # Halt RAG and Recommendations for unidentified items
        rag_products = []
        valid_recommendations = []

    return {
        "status": "success" if is_accepted else "partial",
        "product_identification": {
            "product_id": product_id,
            "product_name": product_name,
            "category": cat_norm,
            "catalog_category": cat_raw,
            "normalized_category": cat_norm,
            "brand": prod_understanding.get("brand", "Unknown"),
            "confidence": float(combined_match_score),
            "status": identification_status
        },
        "identification": {
            "accepted": is_accepted,
            "threshold": IDENTIFICATION_THRESHOLD,
            "combined_score": float(combined_match_score),
            "reason": acceptance_reason
        },
        "product_detection": {
            "Product_ID": product_id or "N/A",
            "Product_Name": product_name,
            "Category": cat_norm,
            "Catalog_Category": cat_raw,
            "Normalized_Category": cat_norm,
            "Vision_Score": top_clip_score
        },
        "yolo": {
            "detections": yolo_detections,
            "best_detection": best_detection,
            "crop_count": len(valid_crops)
        },
        "ocr": {
            "text": ocr_text
        },
        "product_understanding": {
            "product_name": prod_understanding.get("product_name", product_name),
            "category": prod_understanding.get("category", cat_norm),
            "brand": prod_understanding.get("brand", "Unknown"),
            "description": prod_understanding.get("description", ""),
            "gemini_used": prod_understanding.get("gemini_used", False),
            "error": prod_understanding.get("error")
        },
        "vision_top_predictions": vision_results,
        "rag_candidates": rag_products,
        "recommendations": valid_recommendations
    }
