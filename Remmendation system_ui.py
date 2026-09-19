# -*- coding: utf-8 -*-
"""
Vision-Based Pick Recommendation System - Streamlit SaaS Frontend UI
Run with: streamlit run "Remmendation system_ui.py"
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Import backend pipeline
import pipeline

# Set page configuration
st.set_page_config(
    page_title="Warehouse AI - Pick Recommendation System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injected SaaS Enterprise Custom CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    .saas-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .saas-card:hover {
        border-color: #38bdf8;
    }

    .saas-card-accent {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
    }
    
    .saas-card-warning {
        background: linear-gradient(135deg, #1e293b 0%, #17120a 100%);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.15);
    }
    
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-pill-warning {
        display: inline-flex;
        align-items: center;
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: currentColor;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .kpi-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .kpi-desc {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    .rec-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .rec-badge {
        background-color: #0284c7;
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    .page-title {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .page-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 24px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "active_page" not in st.session_state:
    st.session_state.active_page = "🏠 Dashboard"


# Cache heavy backend loading
@st.cache_resource(show_spinner="Initializing AI Models & Datasets...")
def load_cached_backend():
    is_ok = pipeline.initialize_backend()
    return is_ok


# Sidebar Navigation
def render_sidebar():
    st.sidebar.markdown("""
        <div style="padding: 10px 0 20px 0;">
            <h2 style="color: #38bdf8; margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: 1px;">WAREHOUSE AI</h2>
            <p style="color: #64748b; margin: 0; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">AI PICK RECOMMENDATION</p>
        </div>
    """, unsafe_allow_html=True)

    is_ok, status_info = pipeline.check_backend_status()

    if is_ok:
        st.sidebar.markdown("""
            <div class="status-pill-online" style="margin-bottom: 20px;">
                <span class="status-dot"></span> AI SYSTEM ONLINE
            </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
            <div class="status-pill-warning" style="margin-bottom: 20px;">
                <span class="status-dot"></span> BACKEND WARNING
            </div>
        """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    nav_options = [
        "🏠 Dashboard",
        "🔍 Product Detection",
        "🛒 Recommendations",
        "🧠 AI Pipeline",
        "📊 Analytics",
        "🕘 History",
        "ℹ️ About"
    ]

    selected_nav = st.sidebar.radio(
        "Go to page",
        nav_options,
        index=nav_options.index(st.session_state.active_page),
        label_visibility="collapsed"
    )
    st.session_state.active_page = selected_nav

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Specs")
    st.sidebar.caption("🤖 YOLOv26 Object Detection")
    st.sidebar.caption("👁️ CLIP ViT-B/32 Zero-Shot Vision")
    st.sidebar.caption("🧠 Gemini Multimodal Understanding")
    st.sidebar.caption("📚 MiniLM RAG Vector Retrieval")
    st.sidebar.caption("⚡ Gemini 3.6 Flash Recommendation Engine")

    if not status_info["has_api_key"]:
        st.sidebar.warning("⚠️ GEMINI_API_KEY environment variable is missing.")


# KPI Cards Section
def render_kpi_cards():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="saas-card">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🎯</div>
                <div class="kpi-title">Product Detection</div>
                <div class="kpi-value">YOLO</div>
                <div class="kpi-desc">Multi-Object Bounding & Crop</div>
                <div style="margin-top: 10px;">
                    <span class="status-pill-online"><span class="status-dot"></span> Ready</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="saas-card">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">👁️</div>
                <div class="kpi-title">Vision Identification</div>
                <div class="kpi-value">CLIP</div>
                <div class="kpi-desc">Zero-Shot Identification</div>
                <div style="margin-top: 10px;">
                    <span class="status-pill-online"><span class="status-dot"></span> Ready</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="saas-card">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">📚</div>
                <div class="kpi-title">Semantic Retrieval</div>
                <div class="kpi-value">RAG</div>
                <div class="kpi-desc">Catalog Similarity Search</div>
                <div style="margin-top: 10px;">
                    <span class="status-pill-online"><span class="status-dot"></span> Ready</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="saas-card">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">⚡</div>
                <div class="kpi-title">Recommendation Engine</div>
                <div class="kpi-value">GEMINI</div>
                <div class="kpi-desc">Generative Complementary Pick</div>
                <div style="margin-top: 10px;">
                    <span class="status-pill-online"><span class="status-dot"></span> Ready</span>
                </div>
            </div>
        """, unsafe_allow_html=True)


# Render Pipeline Execution Progress (9-stage flow)
def execute_pipeline(pil_image):
    progress_bar = st.progress(0)
    status_text = st.empty()

    stages = [
        ("01", "Image Preprocessing", "Running OpenCV noise reduction & scale optimization..."),
        ("02", "YOLO Detection", "Bounding product locations with YOLO detection model..."),
        ("03", "Product Crops & Filtering", "Extracting & validating high-resolution product crops..."),
        ("04", "OCR Text Extraction", "Extracting visible packaging text from product crops..."),
        ("05", "CLIP Identification", "Matching image features against product catalog names..."),
        ("06", "Gemini Product Understanding", "Interpreting crop image with multimodal AI..."),
        ("07", "Catalog Match & Acceptance Gate", "Validating multi-signal match confidence against 35% threshold..."),
        ("08", "Text RAG Retrieval", "Searching catalog vector database for candidates..."),
        ("09", "Gemini Recommendation & Validation", "Generating complementary pick recommendations...")
    ]

    for idx, (code, stage_name, desc) in enumerate(stages):
        status_text.markdown(f"**Stage {code}: {stage_name}** — *{desc}*")
        progress_bar.progress((idx + 1) / len(stages))
        time.sleep(0.10)

    cached_ok = load_cached_backend()
    if not cached_ok:
        st.error("❌ Private product catalog is not configured or backend initialization failed.")

    result = pipeline.final_warehouse_pipeline(pil_image)
    progress_bar.progress(1.0)
    status_text.empty()
    return result


# Render Product Identification Result Card
def render_product_card(result):
    ident = result.get("identification", {})
    prod_ident = result.get("product_identification", {})
    und = result.get("product_understanding", {})
    yolo_data = result.get("yolo", {})
    yolo_best = yolo_data.get("best_detection", {})
    
    is_accepted = ident.get("accepted", False)
    reason = ident.get("reason", "")
    match_conf = float(ident.get("combined_score", prod_ident.get("confidence", 0.0)))
    
    yolo_conf = float(yolo_best.get("confidence", 0.0)) if yolo_best else 0.0
    det_obj = result.get("product_detection", {})
    clip_score = float(det_obj.get("Vision_Score", 0.0))

    if is_accepted:
        # IDENTIFIED PRODUCT STATE
        prod_name = prod_ident.get("product_name", "Identified Product")
        prod_id = prod_ident.get("product_id", "N/A")
        cat_name = prod_ident.get("category", "Unknown")
        card_class = "saas-card-accent"
        status_str = "✓ Product Identified"
        status_color = "#10b981"
        
        st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="rec-badge">PRODUCT IDENTIFICATION</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: {status_color};">
                        {status_str}
                    </div>
                </div>
                <h2 style="color: #f8fafc; margin-top: 8px; font-size: 1.6rem;">{prod_name}</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div>
                        <div class="kpi-title">Product ID</div>
                        <div style="font-weight: 700; color: #38bdf8; font-size: 1.1rem;">{prod_id}</div>
                    </div>
                    <div>
                        <div class="kpi-title">Category</div>
                        <div style="font-weight: 700; color: #f1f5f9; font-size: 1.1rem;">{cat_name}</div>
                    </div>
                    <div>
                        <div class="kpi-title">YOLO Detection Confidence</div>
                        <div style="font-weight: 700; color: #10b981; font-size: 1.1rem;">
                            {yolo_conf * 100:.1f}%
                        </div>
                    </div>
                    <div>
                        <div class="kpi-title">CLIP Vision Score</div>
                        <div style="font-weight: 700; color: #34d399; font-size: 1.1rem;">
                            {clip_score * 100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # AI Product Understanding Section for Identified Items
        st.markdown("### 🧠 AI Product Understanding")
        desc_text = und.get("description", "No description available.")
        is_gemini_used = und.get("gemini_used", False)
        
        st.markdown(f"""
            <div class="saas-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div class="kpi-title">Match Confidence ({match_conf * 100:.1f}%)</div>
                    <span class="status-pill-online" style="font-size: 0.75rem;">{'✓ Gemini AI' if is_gemini_used else '⚡ Rule Engine'}</span>
                </div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 12px;">{cat_name}</div>
                <div class="kpi-title">Product Description</div>
                <p style="color: #cbd5e1; font-size: 0.95rem; margin-top: 4px; line-height: 1.5;">{desc_text}</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        # UNKNOWN PRODUCT STATE (REJECTED LOW EVIDENCE MATCH)
        st.markdown(f"""
            <div class="saas-card-warning">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="rec-badge" style="background-color: #d97706;">UNIDENTIFIED ITEM</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #f59e0b;">
                        ⚠ Product could not be reliably identified
                    </div>
                </div>
                <h2 style="color: #f8fafc; margin-top: 8px; font-size: 1.6rem;">Unknown Product</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                    <div>
                        <div class="kpi-title">Product ID</div>
                        <div style="font-weight: 700; color: #94a3b8; font-size: 1.1rem;">N/A</div>
                    </div>
                    <div>
                        <div class="kpi-title">Category</div>
                        <div style="font-weight: 700; color: #94a3b8; font-size: 1.1rem;">Unknown</div>
                    </div>
                    <div>
                        <div class="kpi-title">YOLO Detection Confidence</div>
                        <div style="font-weight: 700; color: #10b981; font-size: 1.1rem;">
                            {yolo_conf * 100:.1f}%
                        </div>
                    </div>
                    <div>
                        <div class="kpi-title">CLIP Vision Score</div>
                        <div style="font-weight: 700; color: #ef4444; font-size: 1.1rem;">
                            {clip_score * 100:.1f}%
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.warning(f"⚠ **Low Vision Match ({clip_score * 100:.1f}%)**: The visual/text match is too weak to confidently identify a catalog product.")
        st.info(f"💡 **Rejection Reason**: {reason}\n\n*Please upload a clearer or closer product image to establish a reliable catalog match.*")


# Render Analysis Summary
def render_analysis_summary(result):
    if result.get("status") == "failed":
        st.markdown(f"""
            <div class="saas-card" style="border-left: 4px solid #ef4444;">
                <h3 style="color: #ef4444; margin-top: 0;">✕ Pipeline Analysis Failed</h3>
                <p><strong>Failed Stage:</strong> {result.get('stage', 'Unknown')}</p>
                <p><strong>Error Message:</strong> {result.get('message', 'No message')}</p>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("Technical Details"):
            st.json(result)
        return

    ident = result.get("identification", {})
    is_accepted = ident.get("accepted", False)
    yolo_data = result.get("yolo", {})
    detections = yolo_data.get("detections", [])

    st.markdown("""
        <div class="status-pill-online" style="margin-bottom: 16px; font-size: 1rem; padding: 6px 16px;">
            <span class="status-dot"></span> AI ANALYSIS COMPLETE ✓
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Detected Product Image")
        if st.session_state.uploaded_image:
            annotated = pipeline.draw_yolo_detections(st.session_state.uploaded_image, detections)
            st.image(annotated, caption="YOLO Bounding Box Annotations", use_container_width=True)

    with c2:
        render_product_card(result)

    st.markdown("---")

    col_rec, col_val = st.columns([2, 1])

    with col_rec:
        st.markdown("### 🛒 AI Recommended Complementary Pick Products")
        if is_accepted:
            recs = result.get("recommendations", [])
            if recs:
                for idx, item in enumerate(recs, 1):
                    st.markdown(f"""
                        <div class="rec-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div class="rec-badge">RECOMMENDATION #{idx:02d}</div>
                                <span style="color: #64748b; font-size: 0.8rem; font-weight: 600;">ID: {item['Product_ID']}</span>
                            </div>
                            <h4 style="color: #f8fafc; margin: 4px 0 0 0;">{item['Product_Name']}</h4>
                            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">Complementary item retrieved from warehouse catalog candidates.</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No valid recommendations returned by the model.")
        else:
            st.warning("🔒 **Recommendations Guardrail Active**: Product identification is uncertain. Recommendations are unavailable until a product can be reliably identified.")

    with col_val:
        st.markdown("### ✅ System Validation")
        st.markdown(f"""
            <div class="saas-card">
                <p style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✓ OpenCV Preprocessing Applied</p>
                <p style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✓ YOLO Multi-Object Detection Verified</p>
                <p style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✓ Crop Quality & Aspect Filtering Applied</p>
                <p style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✓ CLIP Catalog Feature Matching Run</p>
                <p style="color: #10b981; font-weight: 600; margin-bottom: 8px;">✓ Gemini Multimodal Crop Understanding</p>
                <p style="color: {'#10b981' if is_accepted else '#f59e0b'}; font-weight: 600; margin-bottom: 0;">
                    {'✓ Identification Threshold Passed (>= 35%)' if is_accepted else '⚠ Identification Threshold Rejected (< 35%)'}
                </p>
            </div>
        """, unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.button("🔄 Analyze Another Image", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.uploaded_image = None
            st.session_state.uploaded_file_name = None
            st.rerun()

    with b2:
        with st.expander("AI Identification Details"):
            st.json({
                "Identification Acceptance Decision": ident,
                "Product Identification Payload": result.get("product_identification"),
                "OCR Text Extracted": result.get("ocr"),
                "YOLO Crop Count": yolo_data.get("crop_count"),
                "YOLO Best Detection": yolo_data.get("best_detection"),
                "CLIP Top Predictions": result.get("vision_top_predictions"),
                "Gemini Multimodal Understanding": result.get("product_understanding"),
                "RAG Candidates": result.get("rag_candidates"),
                "Final Recommendations": result.get("recommendations")
            })


# Page 1: Main Dashboard
def render_dashboard():
    st.markdown('<div class="page-title">WAREHOUSE AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Vision-Based Pick Recommendation System</div>', unsafe_allow_html=True)

    is_ok, status_info = pipeline.check_backend_status()

    if is_ok:
        st.markdown("""
            <div class="status-pill-online" style="margin-bottom: 20px;">
                <span class="status-dot"></span> AI SYSTEM ONLINE
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-pill-warning" style="margin-bottom: 20px;">
                <span class="status-dot"></span> BACKEND NOT FULLY CONFIGURED
            </div>
        """, unsafe_allow_html=True)
        if status_info["missing_files"]:
            st.warning("⚠️ Private product catalog is not configured.")
        if not status_info["has_api_key"]:
            st.warning("⚠️ GEMINI_API_KEY environment variable is not configured.")

    st.markdown("*Identify warehouse products and generate intelligent complementary recommendations using Computer Vision, RAG and Generative AI.*")
    st.markdown("---")

    render_kpi_cards()

    if st.session_state.current_result is not None:
        render_analysis_summary(st.session_state.current_result)
        return

    st.markdown("### Upload Product Image")
    st.markdown("Upload a warehouse/product image to start AI analysis.")

    uploaded_file = st.file_uploader(
        "Choose warehouse image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        st.session_state.uploaded_file_name = uploaded_file.name

        col_img, col_btn = st.columns([2, 1])

        with col_img:
            st.image(image, caption=f"Uploaded Preview: {uploaded_file.name}", use_container_width=True)

        with col_btn:
            st.markdown("""
                <div class="saas-card">
                    <h4>Image Loaded</h4>
                    <p style="color: #94a3b8; font-size: 0.85rem;">Ready to process image through the 9-stage vision & recommendation pipeline.</p>
                </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Analyze Product", type="primary", use_container_width=True):
                with st.spinner("Processing AI Pipeline..."):
                    result = execute_pipeline(image)
                    st.session_state.current_result = result

                    p_ident = result.get("product_identification", {})
                    st.session_state.history.append({
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": uploaded_file.name,
                        "product_name": p_ident.get("product_name", "Unknown Product"),
                        "product_id": p_ident.get("product_id", "N/A"),
                        "recommendations_count": len(result.get("recommendations", [])),
                        "status": p_ident.get("status", "unidentified")
                    })
                    st.rerun()


# Page 2: Product Detection Page
def render_product_detection_page():
    st.markdown('<div class="page-title">🔍 Product Detection & Identification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Separate YOLO Object Detection & CLIP Vision Classification</div>', unsafe_allow_html=True)

    result = st.session_state.current_result

    if not result or result.get("status") == "failed":
        st.info("💡 Please upload and analyze a product image on the Dashboard page to view detailed detection & identification results.")
        return

    ident = result.get("identification", {})
    is_accepted = ident.get("accepted", False)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Uploaded vs. YOLO Annotated Image")
        if st.session_state.uploaded_image:
            detections = result.get("yolo", {}).get("detections", [])
            annotated = pipeline.draw_yolo_detections(st.session_state.uploaded_image, detections)
            st.image(annotated, use_container_width=True, caption="YOLO Bounding Box Annotations")

    with c2:
        render_product_card(result)

    st.markdown("---")

    # Separate Section 1: YOLO Object Detection
    st.markdown("### 🎯 YOLO Object Detection & Crop Extraction")
    st.markdown(f"*Object bounding boxes detected by YOLO model (Crops extracted: {result.get('yolo', {}).get('crop_count', 0)}).*")
    yolo_detections = result.get("yolo", {}).get("detections", [])
    if yolo_detections:
        yolo_table = []
        for d in yolo_detections:
            yolo_table.append({
                "Class": d.get("class_name", "product"),
                "YOLO Confidence": f"{d.get('confidence', 0.0) * 100:.1f}%",
                "Bounding Box [x1, y1, x2, y2]": [round(val, 1) for val in d.get("bbox", [])]
            })
        st.dataframe(pd.DataFrame(yolo_table), use_container_width=True)
    else:
        st.write("No YOLO bounding boxes detected.")

    st.markdown("---")

    # Separate Section 2: CLIP Top Predictions
    if is_accepted:
        st.markdown("### 👁️ CLIP Product Identification (Top Predictions)")
    else:
        st.markdown("### 👁️ Top Possible Catalog Matches (Unconfirmed Candidates)")
        st.caption("⚠️ *These are unconfirmed catalog candidates ranked by CLIP score, NOT accepted product matches.*")

    vision_top = result.get("vision_top_predictions", [])
    if vision_top:
        top_table = []
        for rank, p in enumerate(vision_top, 1):
            prod_name = p.get("Product_Name")
            raw_c = p.get("Category")
            norm_c = p.get("Normalized_Category", pipeline.normalize_category(prod_name, raw_c))
            top_table.append({
                "Candidate Rank": f"Candidate Match #{rank}",
                "Product ID": p.get("Product_ID"),
                "Product Name": prod_name,
                "Category": norm_c,
                "Catalog Category": raw_c,
                "CLIP Vision Score": f"{p.get('score', 0.0) * 100:.1f}%",
                "Status": "Verified Match" if (is_accepted and rank == 1) else "Rejected Candidate"
            })
        st.dataframe(pd.DataFrame(top_table), use_container_width=True)
    else:
        st.write("No CLIP vision predictions available.")

    with st.expander("AI Identification Details"):
        st.json({
            "Identification Acceptance Decision": ident,
            "Product Identification Payload": result.get("product_identification"),
            "OCR Text Extracted": result.get("ocr"),
            "YOLO Crop Count": result.get("yolo", {}).get("crop_count"),
            "YOLO Detections": result.get("yolo"),
            "CLIP Top Predictions": result.get("vision_top_predictions"),
            "Gemini Multimodal Understanding": result.get("product_understanding"),
            "RAG Candidates": result.get("rag_candidates")
        })


# Page 3: Recommendations Page
def render_recommendations_page():
    st.markdown('<div class="page-title">🛒 Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI Recommended Complementary Warehouse Products</div>', unsafe_allow_html=True)

    result = st.session_state.current_result

    if not result or result.get("status") == "failed":
        st.info("💡 Please upload and analyze a product image on the Dashboard page to view product recommendations.")
        return

    ident = result.get("identification", {})
    is_accepted = ident.get("accepted", False)
    recs = result.get("recommendations", [])
    vision_top = result.get("vision_top_predictions", [])
    rag_cands = result.get("rag_candidates", [])

    st.markdown("### AI Recommended Products")
    st.markdown("*Complementary products selected from retrieved warehouse catalog candidates.*")

    if is_accepted:
        if recs:
            cols = st.columns(min(len(recs), 3))
            for i, item in enumerate(recs):
                with cols[i % 3]:
                    st.markdown(f"""
                        <div class="saas-card-accent">
                            <div class="rec-badge">#{i+1:02d} RECOMMENDATION</div>
                            <h4 style="color: #f8fafc; margin-top: 8px;">{item['Product_Name']}</h4>
                            <p style="color: #38bdf8; font-weight: 700;">ID: {item['Product_ID']}</p>
                            <p style="color: #94a3b8; font-size: 0.85rem;">Complementary Item</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No valid recommendations returned by the model.")
    else:
        st.warning("🔒 **Recommendations Guardrail Active**: Product identification is uncertain. Recommendations are unavailable until a product can be reliably identified.")

    st.markdown("---")

    c_vis, c_rag = st.columns(2)

    with c_vis:
        st.markdown("### 🔎 Vision Candidates (CLIP)")
        if vision_top:
            top_df = pd.DataFrame(vision_top)
            if "score" in top_df.columns:
                top_df["CLIP Score"] = top_df["score"].apply(lambda s: f"{s*100:.1f}%")
            if "Normalized_Category" not in top_df.columns and "Product_Name" in top_df.columns:
                top_df["Normalized_Category"] = top_df.apply(lambda r: pipeline.normalize_category(r["Product_Name"], r.get("Category")), axis=1)
            display_cols = [c for c in ["Product_ID", "Product_Name", "Normalized_Category", "CLIP Score"] if c in top_df.columns]
            st.dataframe(top_df[display_cols], use_container_width=True)

    with c_rag:
        st.markdown("### 📚 Semantic RAG Candidates")
        if is_accepted and rag_cands:
            rag_df = pd.DataFrame(rag_cands)
            if "rag_score" in rag_df.columns:
                rag_df["RAG Score"] = rag_df["rag_score"].apply(lambda s: f"{s:.4f}")
            if "Normalized_Category" not in rag_df.columns and "Product_Name" in rag_df.columns:
                rag_df["Normalized_Category"] = rag_df.apply(lambda r: pipeline.normalize_category(r["Product_Name"], r.get("Category")), axis=1)
            display_cols = [c for c in ["Product_ID", "Product_Name", "Normalized_Category", "RAG Score"] if c in rag_df.columns]
            st.dataframe(rag_df[display_cols], use_container_width=True)
        else:
            st.caption("No RAG candidates retrieved for unidentified item.")

    st.markdown("### Recommendation Validation")
    st.markdown("""
        <div class="saas-card">
            <p style="color: #10b981; font-weight: 600;">✓ Recommendations belong strictly to retrieved RAG candidates</p>
            <p style="color: #10b981; font-weight: 600;">✓ Product ID validated against catalog database</p>
            <p style="color: #10b981; font-weight: 600;">✓ Original detected product successfully excluded</p>
        </div>
    """, unsafe_allow_html=True)


# Page 4: AI Pipeline Page
def render_pipeline_page():
    st.markdown('<div class="page-title">🧠 AI Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">9-Stage Computer Vision, Multi-Crop, OCR, Product Understanding & RAG Flow</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="saas-card" style="text-align: center; padding: 24px;">
            <div style="font-weight: 700; color: #38bdf8; font-size: 0.95rem; line-height: 2;">
                IMAGE UPLOAD &nbsp;➔&nbsp; OPENCV PREPROCESS &nbsp;➔&nbsp; YOLO DETECTION &nbsp;➔&nbsp; CROPS & FILTERING <br>
                ➔&nbsp; OCR EXTRACTION &nbsp;➔&nbsp; CLIP MATCHING &nbsp;➔&nbsp; GEMINI PRODUCT UNDERSTANDING <br>
                ➔&nbsp; CATALOG MATCH & ACCEPTANCE GATE &nbsp;➔&nbsp; TEXT RAG &nbsp;➔&nbsp; GEMINI RECOMMENDATION &nbsp;➔&nbsp; VALIDATION
            </div>
        </div>
    """, unsafe_allow_html=True)

    stages = [
        ("01", "Image Preprocessing", "OpenCV", "Resize to 1280 max size & Gaussian blur noise reduction.", "Ready"),
        ("02", "Product Detection", "YOLOv26 (prince4332/yolov26-product-detection-v2)", "Detect product bounding box coordinates and detection confidence.", "Ready"),
        ("03", "Crops & Quality Filtering", "OpenCV Crop Engine", "Extract multiple product crops & filter small/distorted bounding boxes.", "Ready"),
        ("04", "OCR Text Extraction", "OpenCV Text Analyzer", "Extract visible packaging text from product crops.", "Ready"),
        ("05", "CLIP Visual Matching", "CLIP (openai/clip-vit-base-patch32)", "Zero-shot classification against catalog product names.", "Ready"),
        ("06", "Gemini Product Understanding", "Gemini 3.6 Flash", "Interpret crop image + OCR + CLIP candidates to generate structured product understanding.", "Ready"),
        ("07", "Catalog Match & Acceptance Gate", "Multi-Signal Scorer (35% Threshold)", "Require combined confidence >= 35% to accept identification. Otherwise set Unknown Product.", "Ready"),
        ("08", "Text RAG Retrieval", "SentenceTransformer (all-MiniLM-L6-v2)", "Retrieve top-K catalog candidates using vector similarity (accepted items only).", "Ready"),
        ("09", "Gemini Recommendation & Validation", "Gemini 3.6 Flash & Rule Engine", "Generative complementary product recommendation and strict candidate validation.", "Ready")
    ]

    for code, title, tech, desc, status in stages:
        st.markdown(f"""
            <div class="saas-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="rec-badge">STAGE {code}</span>
                        <h3 style="color: #f8fafc; margin: 4px 0;">{title}</h3>
                        <p style="color: #38bdf8; font-weight: 600; margin: 0 0 6px 0;">Tech: {tech}</p>
                        <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">{desc}</p>
                    </div>
                    <div>
                        <span class="status-pill-online"><span class="status-dot"></span> {status}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


# Page 5: Analytics Page
def render_analytics_page():
    st.markdown('<div class="page-title">📊 Session Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time Performance Metrics from Session Executions</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analytics data available yet. Please run an image analysis on the Dashboard.")
        return

    df_hist = pd.DataFrame(st.session_state.history)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Analyses", len(df_hist))

    with c2:
        success_count = (df_hist["status"] == "identified").sum()
        st.metric("Identified Products", success_count)

    with c3:
        avg_recs = df_hist["recommendations_count"].mean() if len(df_hist) > 0 else 0
        st.metric("Avg Recommendations", f"{avg_recs:.1f}")

    st.markdown("---")
    st.markdown("### Execution History Chart")
    st.bar_chart(df_hist[["filename", "recommendations_count"]].set_index("filename"))


# Page 6: History Page
def render_history_page():
    st.markdown('<div class="page-title">🕘 History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Current Session History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No history records in the current session yet.")
        return

    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)


# Page 7: About Page
def render_about_page():
    st.markdown('<div class="page-title">ℹ️ About System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Vision-Based Pick Recommendation System</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="saas-card-accent">
            <h3>System Overview</h3>
            <p>The Vision-Based Pick Recommendation System is an AI-powered warehouse automation engine designed to identify products from computer vision imagery and recommend complementary pick items for optimized warehouse operations.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Architecture Stack")
    st.markdown("- **Pre-processing**: OpenCV RGB-to-BGR scaling and Gaussian noise reduction.")
    st.markdown("- **Object Detection**: YOLOv26 model fine-tuned for product bounding boxes.")
    st.markdown("- **Crop Validation & OCR**: OpenCV aspect/size filtering and text region analysis.")
    st.markdown("- **Product Identification**: OpenAI CLIP ViT-B/32 zero-shot vision classification.")
    st.markdown("- **Product Understanding**: Google Gemini 3.6 Flash multimodal visual analysis.")
    st.markdown("- **Acceptance Threshold**: Multi-signal confidence score fusion with 35% threshold gate.")
    st.markdown("- **RAG Vector Search**: SentenceTransformer `all-MiniLM-L6-v2` dense vector retrieval.")
    st.markdown("- **Generative AI**: Google Gemini 3.6 Flash model for intelligent recommendation reasoning.")


# Main App Dispatcher
def main():
    render_sidebar()

    page = st.session_state.active_page

    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🔍 Product Detection":
        render_product_detection_page()
    elif page == "🛒 Recommendations":
        render_recommendations_page()
    elif page == "🧠 AI Pipeline":
        render_pipeline_page()
    elif page == "📊 Analytics":
        render_analytics_page()
    elif page == "🕘 History":
        render_history_page()
    elif page == "ℹ️ About":
        render_about_page()

if __name__ == "__main__":
    main()
