
import streamlit as st
from PIL import Image
import pandas as pd
import json
import io
from datetime import datetime

# ============================================================
# WAREHOUSE VISION-BASED PICK RECOMMENDATION SYSTEM
# COMPLETE STREAMLIT USER INTERFACE
# ============================================================

st.set_page_config(
    page_title="Warehouse AI | Pick Recommendation",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f6f7fb;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #111827, #374151);
            color: white;
            margin-bottom: 1.4rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.25rem;
        }

        .hero p {
            margin: .55rem 0 0;
            color: #d1d5db;
            font-size: 1rem;
        }

        .card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.2rem;
            margin-bottom: .9rem;
        }

        .small-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem;
            min-height: 120px;
        }

        .number {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }

        .muted {
            color: #6b7280;
            font-size: .9rem;
        }

        .recommendation {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: .75rem;
        }

        .recommendation-number {
            font-size: 1.35rem;
            font-weight: 800;
        }

        .pipeline-step {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 1rem .5rem;
            text-align: center;
            min-height: 105px;
        }

        .pipeline-icon {
            font-size: 1.8rem;
        }

        .success-box {
            padding: .9rem 1rem;
            border-radius: 12px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .warning-box {
            padding: .9rem 1rem;
            border-radius: 12px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 750;
            margin-top: 1rem;
            margin-bottom: .8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state["result"] = None

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

if "history" not in st.session_state:
    st.session_state["history"] = []

# ============================================================
# BACKEND
# ============================================================

@st.cache_resource
def load_backend():
    try:
        from pipeline import final_warehouse_pipeline
        return final_warehouse_pipeline, None
    except Exception as exc:
        return None, str(exc)


final_warehouse_pipeline, backend_error = load_backend()

# ============================================================
# HELPERS
# ============================================================

def reset_analysis():
    st.session_state["result"] = None
    st.session_state["uploaded_image"] = None
    st.session_state["uploaded_filename"] = None


def dataframe_from(value):
    if isinstance(value, list) and value:
        return pd.DataFrame(value)
    return pd.DataFrame()


def result_json(result):
    return json.dumps(result, indent=2, default=str)


def safe_score(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def save_history(result, filename):
    if not result or result.get("status") != "success":
        return

    detected = result.get("product_detection", {})
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": filename or "-",
        "Product_ID": detected.get("Product_ID", "-"),
        "Product_Name": detected.get("Product_Name", "-"),
        "Category": detected.get("Category", "-"),
        "Vision_Score": safe_score(detected.get("Vision_Score", 0)),
        "recommendations": len(result.get("recommendations", [])),
    }

    st.session_state["history"].insert(0, entry)
    st.session_state["history"] = st.session_state["history"][:20]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📦 Warehouse AI")
    st.caption("Vision-Based Pick Recommendation")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Product Detection",
            "🎯 Detection Results",
            "🛒 Recommendations",
            "📚 RAG Explorer",
            "🕘 Run History",
            "📊 Pipeline Details",
        ],
    )

    st.markdown("---")

    st.markdown("### AI Pipeline")

    pipeline_items = [
        ("📷", "Image Upload"),
        ("⚙️", "OpenCV"),
        ("🎯", "YOLO"),
        ("🤖", "CLIP"),
        ("📚", "Text RAG"),
        ("✨", "Gemini"),
        ("✓", "Validation"),
    ]

    for icon, name in pipeline_items:
        st.write(f"{icon} {name}")

    st.markdown("---")

    if final_warehouse_pipeline is not None:
        st.success("Backend connected")
    else:
        st.error("Backend unavailable")

    if st.button("🗑️ Clear Current Analysis", use_container_width=True):
        reset_analysis()
        st.rerun()

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📦 Warehouse Vision-Based Pick Recommendation</h1>
        <p>
            Detect a warehouse product from an image, identify it with vision,
            retrieve relevant catalog candidates and generate validated
            complementary product recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown("## System Overview")

    result = st.session_state["result"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Vision Model", "CLIP")

    with col2:
        st.metric("Object Detection", "YOLO")

    with col3:
        st.metric("Retrieval", "Text RAG")

    with col4:
        st.metric("Recommendation", "Gemini")

    st.markdown("---")

    st.markdown("## End-to-End Workflow")

    steps = [
        ("📷", "Upload", "Product image"),
        ("⚙️", "Preprocess", "OpenCV"),
        ("🎯", "Detect", "YOLO"),
        ("🤖", "Identify", "CLIP"),
        ("📚", "Retrieve", "Text RAG"),
        ("✨", "Recommend", "Gemini"),
        ("✓", "Validate", "Candidate check"),
    ]

    cols = st.columns(len(steps))

    for col, (icon, title, subtitle) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-icon">{icon}</div>
                    <b>{title}</b><br>
                    <span class="muted">{subtitle}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown("## Current Analysis")

    if result and result.get("status") == "success":
        detected = result.get("product_detection", {})
        recommendations = result.get("recommendations", [])
        rag = result.get("rag_candidates", [])
        detections = result.get("yolo", {}).get("detections", [])

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                f'<div class="small-card"><p class="number">{len(detections)}</p>'
                '<span class="muted">YOLO detections</span></div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f'<div class="small-card"><p class="number">{len(rag)}</p>'
                '<span class="muted">RAG candidates</span></div>',
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f'<div class="small-card"><p class="number">{len(recommendations)}</p>'
                '<span class="muted">Recommendations</span></div>',
                unsafe_allow_html=True,
            )

        with c4:
            st.markdown(
                f'<div class="small-card"><p class="number">{safe_score(detected.get("Vision_Score", 0)):.2f}</p>'
                '<span class="muted">Vision score</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Detected Product")

        a, b = st.columns([1, 2])

        with a:
            if st.session_state["uploaded_image"] is not None:
                st.image(
                    st.session_state["uploaded_image"],
                    use_container_width=True,
                )

        with b:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{detected.get("Product_Name", "Unknown Product")}</h3>
                    <p><b>Product ID:</b> {detected.get("Product_ID", "-")}</p>
                    <p><b>Category:</b> {detected.get("Category", "-")}</p>
                    <p><b>Vision Score:</b>
                       {safe_score(detected.get("Vision_Score", 0)):.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.info(
            "No completed analysis yet. Open Product Detection, upload an image, "
            "and run the AI pipeline."
        )

# ============================================================
# PRODUCT DETECTION
# ============================================================

elif page == "🔍 Product Detection":

    st.markdown("## Product Detection")

    st.markdown(
        """
        <div class="card">
            <b>Step 1:</b> Upload a product image.<br>
            <b>Step 2:</b> Review the image information.<br>
            <b>Step 3:</b> Run the complete YOLO → CLIP → RAG → Gemini pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a warehouse/product image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an image containing the product you want to identify.",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.session_state["uploaded_image"] = image
        st.session_state["uploaded_filename"] = uploaded_file.name

        left, right = st.columns([1.35, 1])

        with left:
            st.markdown("### Preview")
            st.image(image, use_container_width=True)

        with right:
            st.markdown("### Image Information")

            st.markdown(
                f"""
                <div class="card">
                    <p><b>File:</b> {uploaded_file.name}</p>
                    <p><b>Width:</b> {image.width}px</p>
                    <p><b>Height:</b> {image.height}px</p>
                    <p><b>Format:</b> {image.format or "RGB image"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Pipeline Settings")

            top_k_vision = st.slider(
                "Top vision predictions",
                min_value=1,
                max_value=10,
                value=5,
            )

            top_k_rag = st.slider(
                "Top RAG candidates",
                min_value=1,
                max_value=20,
                value=10,
            )

            run = st.button(
                "🚀 Run Complete AI Pipeline",
                type="primary",
                use_container_width=True,
            )

            if run:

                if final_warehouse_pipeline is None:
                    st.error(
                        "Backend pipeline was not found. "
                        "Make sure pipeline.py is in the same folder."
                    )

                else:
                    with st.status(
                        "Running AI pipeline...",
                        expanded=True,
                    ) as status:

                        try:
                            st.write("⚙️ Running image preprocessing...")
                            st.write("🎯 Running YOLO product detection...")
                            st.write("🤖 Running CLIP product identification...")
                            st.write("📚 Running Text RAG retrieval...")
                            st.write("✨ Generating Gemini recommendations...")

                            result = final_warehouse_pipeline(
                                image,
                                top_k_vision=top_k_vision,
                                top_k_rag=top_k_rag,
                            )

                            st.session_state["result"] = result

                            if result.get("status") == "success":
                                save_history(result, uploaded_file.name)
                                status.update(
                                    label="Pipeline completed successfully",
                                    state="complete",
                                )
                            else:
                                status.update(
                                    label="Pipeline failed",
                                    state="error",
                                )

                        except Exception as exc:
                            status.update(
                                label="Pipeline error",
                                state="error",
                            )
                            st.exception(exc)

    else:
        st.info("Upload a product image to begin.")

# ============================================================
# DETECTION RESULTS
# ============================================================

elif page == "🎯 Detection Results":

    st.markdown("## Detection Results")

    result = st.session_state["result"]
    image = st.session_state["uploaded_image"]

    if not result:
        st.info(
            "No analysis result is available. Run the pipeline from "
            "Product Detection first."
        )

    elif result.get("status") != "success":
        st.error(result.get("message", "Pipeline failed."))

        if result.get("stage"):
            st.write(f"**Failed stage:** {result.get('stage')}")

        if result.get("error"):
            st.code(str(result.get("error")))

    else:

        detected = result.get("product_detection", {})
        detections = result.get("yolo", {}).get("detections", [])
        best_detection = result.get("yolo", {}).get("best_detection")

        st.markdown("### Detected Product")

        a, b = st.columns([1, 2])

        with a:
            if image is not None:
                st.image(image, use_container_width=True)

        with b:
            st.markdown(
                f"""
                <div class="card">
                    <h2>{detected.get("Product_Name", "Unknown")}</h2>
                    <p><b>Product ID:</b> {detected.get("Product_ID", "-")}</p>
                    <p><b>Category:</b> {detected.get("Category", "-")}</p>
                    <p><b>Vision Score:</b>
                       {safe_score(detected.get("Vision_Score", 0)):.4f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("### 🎯 YOLO Detection")

        if best_detection:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Best Class",
                    best_detection.get("class_name", "-"),
                )

            with c2:
                st.metric(
                    "Confidence",
                    f"{safe_score(best_detection.get('confidence', 0)):.4f}",
                )

            with c3:
                st.metric(
                    "Detection Count",
                    len(detections),
                )

        if detections:
            df = dataframe_from(detections)

            if "confidence" in df.columns:
                df["confidence"] = df["confidence"].apply(
                    lambda x: round(safe_score(x), 4)
                )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("No YOLO detections were returned.")

        st.markdown("---")

        st.markdown("### 🤖 Top Vision Predictions")

        vision = result.get("vision_top_predictions", [])

        if vision:
            df = dataframe_from(vision)

            if "score" in df.columns:
                df["score"] = df["score"].apply(
                    lambda x: round(safe_score(x), 4)
                )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No vision predictions were returned.")

# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🛒 Recommendations":

    st.markdown("## 🛒 AI Product Recommendations")

    result = st.session_state["result"]
    image = st.session_state["uploaded_image"]

    if not result:
        st.info(
            "Run Product Detection first to generate recommendations."
        )

    elif result.get("status") != "success":
        st.error(result.get("message", "Pipeline failed."))

    else:

        detected = result.get("product_detection", {})
        recommendations = result.get("recommendations", [])
        rag_candidates = result.get("rag_candidates", [])

        a, b = st.columns([1, 2])

        with a:
            if image is not None:
                st.image(image, use_container_width=True)

        with b:
            st.markdown(
                f"""
                <div class="card">
                    <p class="muted">Detected product</p>
                    <h2>{detected.get("Product_Name", "Unknown")}</h2>
                    <p><b>Product ID:</b> {detected.get("Product_ID", "-")}</p>
                    <p><b>Category:</b> {detected.get("Category", "-")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("RAG Candidates", len(rag_candidates))

        with c2:
            st.metric("Final Recommendations", len(recommendations))

        with c3:
            st.metric(
                "Vision Score",
                f"{safe_score(detected.get('Vision_Score', 0)):.4f}",
            )

        st.markdown("### Recommended Products")

        if recommendations:
            for index, product in enumerate(recommendations, start=1):

                st.markdown(
                    f"""
                    <div class="recommendation">
                        <span class="recommendation-number">
                            {index}. {product.get("Product_Name", "Unknown")}
                        </span>
                        <br>
                        <span class="muted">
                            Product ID: {product.get("Product_ID", "-")}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No valid recommendations were generated.")

        st.markdown("---")

        st.markdown("### Validation")

        rag_ids = {
            str(p.get("Product_ID"))
            for p in rag_candidates
        }

        detected_id = str(detected.get("Product_ID"))

        valid_count = 0
        invalid_count = 0

        for product in recommendations:
            product_id = str(product.get("Product_ID"))
            if product_id in rag_ids and product_id != detected_id:
                valid_count += 1
            else:
                invalid_count += 1

        if invalid_count == 0:
            st.success(
                f"All {valid_count} displayed recommendations passed "
                "the candidate validation check."
            )
        else:
            st.warning(
                f"{valid_count} recommendations passed validation and "
                f"{invalid_count} did not."
            )

        # Download the final recommendation list.
        recommendation_json = json.dumps(
            recommendations,
            indent=2,
            default=str,
        )

        st.download_button(
            "⬇️ Download Recommendations JSON",
            data=recommendation_json,
            file_name="warehouse_recommendations.json",
            mime="application/json",
        )

# ============================================================
# RAG EXPLORER
# ============================================================

elif page == "📚 RAG Explorer":

    st.markdown("## 📚 RAG Candidate Explorer")

    result = st.session_state["result"]

    if not result or result.get("status") != "success":
        st.info(
            "Run the AI pipeline first to view the retrieved RAG candidates."
        )

    else:

        rag_candidates = result.get("rag_candidates", [])

        if not rag_candidates:
            st.warning("No RAG candidates were returned.")

        else:

            df = dataframe_from(rag_candidates)

            if "rag_score" in df.columns:
                df["rag_score"] = df["rag_score"].apply(
                    lambda x: round(safe_score(x), 4)
                )

            search = st.text_input(
                "Search candidate products",
                placeholder="Search Product ID, name or category...",
            )

            if search:
                mask = (
                    df.astype(str)
                    .apply(
                        lambda row: row.str.contains(
                            search,
                            case=False,
                            na=False,
                        ).any(),
                        axis=1,
                    )
                )
                display_df = df[mask]
            else:
                display_df = df

            st.metric("Candidates shown", len(display_df))

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=False)

            st.download_button(
                "⬇️ Download RAG Candidates CSV",
                data=csv_buffer.getvalue(),
                file_name="rag_candidates.csv",
                mime="text/csv",
            )

# ============================================================
# RUN HISTORY
# ============================================================

elif page == "🕘 Run History":

    st.markdown("## 🕘 Analysis History")

    history = st.session_state["history"]

    if not history:
        st.info(
            "No successful pipeline runs have been recorded in this "
            "Streamlit session."
        )

    else:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)

        st.download_button(
            "⬇️ Download Run History CSV",
            data=csv_buffer.getvalue(),
            file_name="warehouse_run_history.csv",
            mime="text/csv",
        )

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

# ============================================================
# PIPELINE DETAILS
# ============================================================

elif page == "📊 Pipeline Details":

    st.markdown("## 📊 Pipeline Details")

    st.markdown(
        """
        <div class="card">
            <h3>1. Image Input</h3>
            The user uploads a product image through the Streamlit interface.

            <h3>2. OpenCV Preprocessing</h3>
            The image is converted and resized when required before detection.

            <h3>3. YOLO Product Detection</h3>
            The pretrained YOLO model detects the product region and returns
            bounding boxes and confidence values.

            <h3>4. Product Identification</h3>
            The detected product crop is compared with product catalog names
            using the pretrained CLIP vision model.

            <h3>5. Text RAG</h3>
            Product catalog information is converted into text documents.
            Sentence Transformer embeddings retrieve relevant candidate products.

            <h3>6. Gemini Recommendation</h3>
            Gemini receives the detected product and RAG candidates and returns
            product recommendations.

            <h3>7. Validation</h3>
            The final recommendations are checked against the RAG candidate list
            so unknown Product IDs or invented products are not accepted.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("## Backend Connection")

    if final_warehouse_pipeline is not None:
        st.success(
            "Backend pipeline connected successfully."
        )
    else:
        st.error(
            "Backend pipeline is not connected."
        )

        if backend_error:
            with st.expander("Technical error"):
                st.code(backend_error)

    st.markdown("---")

    st.markdown("## Backend Contract")

    st.code(
        """
final_warehouse_pipeline(
    pil_image,
    top_k_vision=5,
    top_k_rag=10
)
        """,
        language="python",
    )

    st.markdown("## Expected Result Structure")

    st.code(
        """
{
    "status": "success",

    "yolo": {
        "detections": [...],
        "best_detection": {...}
    },

    "product_detection": {
        "Product_ID": "...",
        "Product_Name": "...",
        "Category": "...",
        "Vision_Score": 0.0
    },

    "vision_top_predictions": [...],
    "rag_candidates": [...],
    "recommendations": [...]
}
        """,
        language="python",
    )

    st.markdown("---")

    st.markdown("## Current Result JSON")

    current_result = st.session_state["result"]

    if current_result:
        st.json(current_result)

        st.download_button(
            "⬇️ Download Complete Result JSON",
            data=result_json(current_result),
            file_name="warehouse_pipeline_result.json",
            mime="application/json",
        )
    else:
        st.info("No result is available yet.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Warehouse Vision-Based Pick Recommendation System | "
    "YOLO + CLIP + Text RAG + Gemini"
)
