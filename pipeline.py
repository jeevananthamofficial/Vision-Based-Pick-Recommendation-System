# ============================================================
# WAREHOUSE VISION-BASED PRODUCT RECOMMENDATION SYSTEM
# BACKEND PIPELINE
# ============================================================
#
# This file contains the AI/backend logic used by warehouse_ui.py.
#
# Flow:
# Image -> OpenCV -> YOLO -> CLIP -> Text RAG -> Gemini
#      -> Validated Product Recommendations
#
# Required project files:
#   cv_images.csv
#   inventory.csv
#   orders.csv
#   warehouse_locations.csv
#
# Recommended folder structure:
#
# warehouse_project/
# ├── warehouse_ui.py
# ├── pipeline.py
# ├── requirements.txt
# ├── cv_images.csv
# ├── inventory.csv
# ├── orders.csv
# └── warehouse_locations.csv
#
# Set your Gemini API key before running:
# Windows PowerShell:
#   $env:GEMINI_API_KEY="YOUR_KEY"
#
# Linux/macOS:
#   export GEMINI_API_KEY="YOUR_KEY"
# ============================================================

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
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification
)
from google import genai


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = os.environ.get(
    "WAREHOUSE_PROJECT_DIR",
    os.path.dirname(os.path.abspath(__file__))
)

CV_IMAGES_PATH = os.path.join(PROJECT_DIR, "cv_images.csv")
INVENTORY_PATH = os.path.join(PROJECT_DIR, "inventory.csv")
ORDERS_PATH = os.path.join(PROJECT_DIR, "orders.csv")
WAREHOUSE_PATH = os.path.join(
    PROJECT_DIR,
    "warehouse_locations.csv"
)

HF_REPO_ID = "prince4332/yolov26-product-detection-v2"
MODEL_FILENAME = "best.pt"

VISION_MODEL = "openai/clip-vit-base-patch32"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. LOAD CSV DATA
# ============================================================

def load_project_data():

    cv_images_df = pd.read_csv(CV_IMAGES_PATH)
    inventory_df = pd.read_csv(INVENTORY_PATH)
    orders_df = pd.read_csv(ORDERS_PATH)
    warehouse_df = pd.read_csv(WAREHOUSE_PATH)

    return (
        cv_images_df,
        inventory_df,
        orders_df,
        warehouse_df
    )


# ============================================================
# 3. BUILD PRODUCT CATALOG
# ============================================================

def build_product_catalog(orders_df):

    required_columns = [
        "Product_ID",
        "Product_Name"
    ]

    for col in required_columns:
        if col not in orders_df.columns:
            raise ValueError(
                f"Required column '{col}' is missing "
                "from orders.csv"
            )

    catalog_columns = [
        "Product_ID",
        "Product_Name",
        "Category"
    ]

    available_columns = [
        c for c in catalog_columns
        if c in orders_df.columns
    ]

    product_catalog = (
        orders_df[available_columns]
        .drop_duplicates("Product_ID")
        .reset_index(drop=True)
    )

    if "Category" not in product_catalog.columns:
        product_catalog["Category"] = "Unknown"

    product_catalog["Product_ID"] = (
        product_catalog["Product_ID"]
        .astype(str)
        .str.strip()
    )

    product_catalog["Product_Name"] = (
        product_catalog["Product_Name"]
        .astype(str)
        .str.strip()
    )

    product_catalog["Category"] = (
        product_catalog["Category"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    return product_catalog


# ============================================================
# 4. LOAD MODELS AND DATA
# ============================================================

def initialize_system():

    (
        cv_images_df,
        inventory_df,
        orders_df,
        warehouse_df
    ) = load_project_data()

    product_catalog = build_product_catalog(
        orders_df
    )

    product_documents = [
        f"""
Product ID: {row['Product_ID']}
Product Name: {row['Product_Name']}
Category: {row['Category']}
""".strip()
        for _, row in product_catalog.iterrows()
    ]

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    product_embeddings = embedding_model.encode(
        product_documents,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    product_embeddings = np.asarray(
        product_embeddings
    )

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    yolo_model = YOLO(model_path)

    vision_processor = AutoProcessor.from_pretrained(
        VISION_MODEL
    )

    vision_model = (
        AutoModelForZeroShotImageClassification
        .from_pretrained(VISION_MODEL)
    )

    vision_model = vision_model.to(DEVICE)
    vision_model.eval()

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable "
            "is not set."
        )

    gemini_client = genai.Client(
        api_key=api_key
    )

    return {
        "cv_images_df": cv_images_df,
        "inventory_df": inventory_df,
        "orders_df": orders_df,
        "warehouse_df": warehouse_df,
        "product_catalog": product_catalog,
        "product_documents": product_documents,
        "embedding_model": embedding_model,
        "product_embeddings": product_embeddings,
        "yolo_model": yolo_model,
        "vision_processor": vision_processor,
        "vision_model": vision_model,
        "gemini_client": gemini_client
    }


# ============================================================
# 5. RAG RETRIEVAL
# ============================================================

def retrieve_products(
    query,
    system,
    top_k=10
):

    embedding_model = system[
        "embedding_model"
    ]

    product_embeddings = system[
        "product_embeddings"
    ]

    product_catalog = system[
        "product_catalog"
    ]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = (
        product_embeddings
        @ query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        row = product_catalog.iloc[idx]

        results.append({
            "Product_ID": str(
                row["Product_ID"]
            ),
            "Product_Name": str(
                row["Product_Name"]
            ),
            "Category": str(
                row["Category"]
            ),
            "rag_score": float(
                scores[idx]
            )
        })

    return results


# ============================================================
# 6. OPENCV PREPROCESSING
# ============================================================

def preprocess_image_opencv(pil_image):

    img = np.array(pil_image)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    max_size = 1280

    height, width = img.shape[:2]

    scale = min(
        max_size / width,
        max_size / height,
        1.0
    )

    if scale < 1.0:

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        img = cv2.resize(
            img,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    return img


# ============================================================
# 7. YOLO DETECTION
# ============================================================

def detect_with_yolo(
    pil_image,
    system,
    confidence=0.25
):

    processed = preprocess_image_opencv(
        pil_image
    )

    yolo_model = system[
        "yolo_model"
    ]

    results = yolo_model.predict(
        source=processed,
        conf=confidence,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence_value = float(
            box.conf[0].item()
        )

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        detections.append({
            "class_id": class_id,
            "class_name": yolo_model.names[
                class_id
            ],
            "confidence": confidence_value,
            "bbox": [
                float(x)
                for x in xyxy
            ]
        })

    return detections


# ============================================================
# 8. YOLO PRODUCT CROP
# ============================================================

def get_yolo_product_crop(
    pil_image,
    system,
    confidence_threshold=0.25
):

    yolo_model = system[
        "yolo_model"
    ]

    results = yolo_model.predict(
        source=np.array(pil_image),
        conf=confidence_threshold,
        iou=0.45,
        verbose=False
    )

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return None, []

    detections = []

    best_box = None
    best_confidence = 0

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        detection = {
            "class_id": class_id,
            "class_name": yolo_model.names[
                class_id
            ],
            "confidence": confidence,
            "bbox": [
                float(x)
                for x in xyxy
            ]
        }

        detections.append(
            detection
        )

        if confidence > best_confidence:

            best_confidence = confidence
            best_box = xyxy

    if best_box is None:
        return None, detections

    x1, y1, x2, y2 = map(
        int,
        best_box
    )

    width, height = pil_image.size

    x1 = max(
        0,
        min(x1, width)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None, detections

    cropped_image = pil_image.crop(
        (x1, y1, x2, y2)
    )

    return (
        cropped_image,
        detections
    )


# ============================================================
# 9. CLIP PRODUCT IDENTIFICATION
# ============================================================

def identify_product_from_image(
    image,
    system,
    top_k=5
):

    product_catalog = system[
        "product_catalog"
    ]

    product_names = (
        product_catalog[
            "Product_Name"
        ].tolist()
    )

    labels = [
        f"a photo of {name}"
        for name in product_names
    ]

    vision_processor = system[
        "vision_processor"
    ]

    vision_model = system[
        "vision_model"
    ]

    inputs = vision_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = vision_model(
            **inputs
        )

    logits = outputs.logits_per_image

    probabilities = (
        logits
        .softmax(dim=1)
        .cpu()
        .numpy()[0]
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        row = product_catalog.iloc[idx]

        results.append({
            "Product_ID": str(
                row["Product_ID"]
            ),
            "Product_Name": str(
                row["Product_Name"]
            ),
            "Category": str(
                row["Category"]
            ),
            "score": float(
                probabilities[idx]
            )
        })

    return results


# ============================================================
# 10. GEMINI RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    detected_product,
    rag_products,
    system,
    number_of_recommendations=5
):

    candidate_text = "\n".join(
        [
            f"""
Product_ID: {p['Product_ID']}
Product_Name: {p['Product_Name']}
Category: {p['Category']}
RAG_Score: {p['rag_score']:.4f}
""".strip()
            for p in rag_products
        ]
    )

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

    client = system[
        "gemini_client"
    ]

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    text = interaction.output_text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    data = json.loads(text)

    valid_products = {
        p["Product_ID"]:
            p["Product_Name"]
        for p in rag_products
    }

    detected_id = (
        detected_product[
            "Product_ID"
        ]
    )

    final_recommendations = []

    for item in data.get(
        "recommendations",
        []
    ):

        product_id = str(
            item.get(
                "Product_ID",
                ""
            )
        ).strip()

        if product_id not in valid_products:
            continue

        if product_id == detected_id:
            continue

        final_recommendations.append({
            "Product_ID": product_id,
            "Product_Name":
                valid_products[product_id]
        })

    unique = {}

    for item in final_recommendations:

        unique[
            item["Product_ID"]
        ] = item

    return {
        "recommendations":
            list(
                unique.values()
            )[
                :number_of_recommendations
            ]
    }


# ============================================================
# 11. DRAW YOLO DETECTIONS
# ============================================================

def draw_yolo_detections(
    pil_image,
    detections
):

    img = np.array(
        pil_image
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    for detection in detections:

        x1, y1, x2, y2 = map(
            int,
            detection["bbox"]
        )

        class_name = (
            detection["class_name"]
        )

        confidence = (
            detection["confidence"]
        )

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            label,
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img


# ============================================================
# 12. COMPLETE WAREHOUSE PIPELINE
# ============================================================

def final_warehouse_pipeline(
    pil_image,
    top_k_vision=5,
    top_k_rag=10
):

    # --------------------------------------------------------
    # STEP A - YOLO DETECTION
    # --------------------------------------------------------

    product_crop, yolo_detections = (
        get_yolo_product_crop(
            pil_image,
            system=SYSTEM
        )
    )

    if product_crop is None:

        return {
            "status": "failed",
            "stage": "YOLO",
            "message":
                "YOLO could not detect a product.",
            "yolo_detections": []
        }

    # --------------------------------------------------------
    # STEP B - PRODUCT IDENTIFICATION
    # --------------------------------------------------------

    vision_results = (
        identify_product_from_image(
            product_crop,
            system=SYSTEM,
            top_k=top_k_vision
        )
    )

    if not vision_results:

        return {
            "status": "failed",
            "stage": "VISION",
            "message":
                "Product identification failed.",
            "yolo_detections":
                yolo_detections
        }

    detected_product = (
        vision_results[0]
    )

    # --------------------------------------------------------
    # STEP C - TEXT RAG
    # --------------------------------------------------------

    rag_query = (
        f"{detected_product['Product_Name']} "
        f"{detected_product['Category']}"
    )

    rag_products = retrieve_products(
        rag_query,
        system=SYSTEM,
        top_k=top_k_rag
    )

    rag_products = [
        p
        for p in rag_products
        if p["Product_ID"]
        != detected_product["Product_ID"]
    ]

    # --------------------------------------------------------
    # STEP D - GEMINI RECOMMENDATION
    # --------------------------------------------------------

    recommendation_result = (
        generate_recommendations(
            detected_product=
                detected_product,
            rag_products=
                rag_products,
            system=SYSTEM,
            number_of_recommendations=5
        )
    )

    recommendations = (
        recommendation_result.get(
            "recommendations",
            []
        )
    )

    # --------------------------------------------------------
    # STEP E - FINAL VALIDATION
    # --------------------------------------------------------

    rag_ids = {
        p["Product_ID"]
        for p in rag_products
    }

    valid_recommendations = [
        p
        for p in recommendations
        if (
            p["Product_ID"]
            in rag_ids
        )
        and (
            p["Product_ID"]
            != detected_product[
                "Product_ID"
            ]
        )
    ]

    return {
        "status": "success",

        "yolo": {
            "detections":
                yolo_detections,

            "best_detection":
                yolo_detections[0]
                if yolo_detections
                else None
        },

        "product_detection": {
            "Product_ID":
                detected_product[
                    "Product_ID"
                ],

            "Product_Name":
                detected_product[
                    "Product_Name"
                ],

            "Category":
                detected_product[
                    "Category"
                ],

            "Vision_Score":
                detected_product.get(
                    "score",
                    0
                )
        },

        "vision_top_predictions":
            vision_results,

        "rag_candidates":
            rag_products,

        "recommendations":
            valid_recommendations
    }


# ============================================================
# 13. INITIALIZE SYSTEM
# ============================================================

# Streamlit caches this module through the UI process.
# Models are loaded once when the backend is imported.

try:
    SYSTEM = initialize_system()
except Exception as exc:
    SYSTEM = None
    INITIALIZATION_ERROR = str(exc)


# ============================================================
# 14. SAFE PIPELINE WRAPPER
# ============================================================

_original_pipeline = final_warehouse_pipeline


def final_warehouse_pipeline(
    pil_image,
    top_k_vision=5,
    top_k_rag=10
):

    if SYSTEM is None:

        return {
            "status": "failed",
            "stage": "INITIALIZATION",
            "message":
                "AI system initialization failed.",
            "error":
                INITIALIZATION_ERROR
        }

    return _original_pipeline(
        pil_image,
        top_k_vision=top_k_vision,
        top_k_rag=top_k_rag
    )
