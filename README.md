# 📦 Vision-Based Pick Recommendation System

## 🚀 Project Overview

The **Warehouse Vision-Based Pick Recommendation System** is an
AI-powered application designed to identify products from warehouse
images and generate complementary product recommendations.

The project combines **computer vision, semantic search,
Retrieval-Augmented Generation (RAG), and Generative AI** into one
end-to-end workflow.

### Core Flow

``` text
                    ┌─────────────────────┐
                    │   Product Image     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ OpenCV Preprocessing│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  YOLO Detection     │
                    │ Product Localization│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ CLIP Identification │
                    │ Product Matching    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Text RAG         │
                    │ Candidate Retrieval │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Gemini AI          │
                    │ Recommendations     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Recommendation      │
                    │ Validation           │
                    └─────────────────────┘
```

------------------------------------------------------------------------

# 🎯 Objectives

The main objectives of the project are:

-   Detect products from warehouse/product images.
-   Locate the product using YOLO object detection.
-   Identify the product using CLIP vision-language matching.
-   Retrieve semantically related products from the product catalog.
-   Generate complementary recommendations using Gemini.
-   Validate recommendations against real catalog Product IDs.
-   Provide an interactive Streamlit interface for users.

------------------------------------------------------------------------

# 🧠 AI Pipeline

The project follows a multi-stage AI pipeline.

## 1. Image Upload

The user uploads a product image through the Streamlit interface.

``` text
User
 ↓
Upload Product Image
```

The image is loaded as a PIL RGB image and passed to the backend.

------------------------------------------------------------------------

## 2. OpenCV Image Preprocessing

OpenCV is used to prepare the image for computer vision processing.

Main operations include:

-   RGB/BGR conversion
-   Image resizing
-   Gaussian blur/noise reduction

Large images are resized to a maximum dimension of approximately 1280
pixels.

``` text
Input Image
    ↓
OpenCV
    ↓
Resize
    ↓
Noise Reduction
    ↓
Processed Image
```

------------------------------------------------------------------------

## 3. YOLO Product Detection

The YOLO model detects products in the image.

The detection stage returns:

-   Class ID
-   Class name
-   Confidence score
-   Bounding box coordinates

The highest-confidence detection is used to extract the main product
crop.

``` text
Processed Image
       ↓
      YOLO
       ↓
 ┌───────────────┐
 │ Bounding Box  │
 │ Confidence    │
 │ Product Class │
 └───────────────┘
       ↓
 Product Crop
```

The project uses a pretrained YOLO model downloaded through Hugging Face
Hub.

------------------------------------------------------------------------

## 4. CLIP Product Identification

After YOLO extracts the product region, CLIP compares the image with
product names from the catalog.

Product names are converted into prompts such as:

``` text
a photo of <Product Name>
```

CLIP calculates image-text similarity scores and returns the top product
predictions.

Output includes:

``` text
Product_ID
Product_Name
Category
Vision_Score
```

Example:

``` text
Product ID   : P001
Product Name : Wireless Mouse
Category     : Electronics
Vision Score : 0.94
```

------------------------------------------------------------------------

# 📚 5. Text RAG Retrieval

The project creates a text document for every product in the catalog.

Example:

``` text
Product ID: P001
Product Name: Wireless Mouse
Category: Electronics
```

These documents are embedded using:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The system creates normalized embeddings and performs semantic
similarity retrieval.

``` text
Product Catalog
      ↓
Text Documents
      ↓
Sentence Transformer
      ↓
Product Embeddings
      ↓
Similarity Search
      ↓
Top RAG Candidates
```

The detected product name and category are used as the retrieval query.

------------------------------------------------------------------------

# ✨ 6. Gemini Recommendation Generation

The retrieved RAG candidates are sent to Gemini.

Gemini is instructed to:

1.  Recommend complementary products.
2.  Use only products supplied by RAG.
3.  Avoid the detected product.
4.  Never invent Product IDs.
5.  Preserve the original Product ID.
6.  Preserve the original Product Name.
7.  Return structured JSON.

Expected structure:

``` json
{
  "recommendations": [
    {
      "Product_ID": "P0001",
      "Product_Name": "Example Product"
    }
  ]
}
```

------------------------------------------------------------------------

# ✅ 7. Recommendation Validation

The final recommendations are validated against the RAG candidate
Product IDs.

The validation ensures:

``` text
Generated Product
       ↓
Is Product ID in RAG candidates?
       │
    ┌──┴──┐
   YES    NO
    │      │
    ▼      ▼
 Accept   Reject
```

The detected product is also excluded from the final recommendations.

------------------------------------------------------------------------

# 🖥️ Streamlit User Interface

The project includes a Streamlit-based interface.

The UI provides sections for:

-   🏠 Dashboard
-   🔍 Product Detection
-   🛒 Recommendations
-   📊 Pipeline Details

The dashboard explains the complete AI workflow and allows the user to
interact with the backend pipeline.

### UI Flow

``` text
Streamlit
    │
    ├── Upload Image
    │
    ├── Run AI Pipeline
    │
    ├── Product Detection
    │
    ├── Vision Predictions
    │
    ├── RAG Candidates
    │
    └── AI Recommendations
```

------------------------------------------------------------------------

# 📁 Project Structure

``` text
warehouse_vision_project/
│
├── pipeline.py
├── Remmendation system_ui.py
├── warehouse_ui.py
├── requirements.txt
├── README.md
│
├── cv_images.csv
├── inventory.csv
├── orders.csv
└── warehouse_locations.csv
```

### Main Files

  File                          Purpose
  ----------------------------- ------------------------------
  `pipeline.py`                 Complete AI/backend pipeline
  `Remmendation system_ui.py`   Streamlit recommendation UI
  `warehouse_ui.py`             Alternative Streamlit UI
  `cv_images.csv`               Computer vision image data
  `inventory.csv`               Inventory data
  `orders.csv`                  Product/order catalog
  `warehouse_locations.csv`     Warehouse location data
  `requirements.txt`            Python dependencies

------------------------------------------------------------------------

# 🗂️ Dataset

The backend expects these four CSV files:

``` text
cv_images.csv
inventory.csv
orders.csv
warehouse_locations.csv
```

The product catalog is constructed from product information including:

``` text
Product_ID
Product_Name
Category
```

The original project dataset should be used so that the vision
identification, RAG retrieval, and recommendation stages operate on real
catalog information.

------------------------------------------------------------------------

# 🛠️ Technology Stack

  Layer                   Technology
  ----------------------- -----------------------
  Programming Language    Python
  Web Interface           Streamlit
  Image Processing        OpenCV
  Object Detection        Ultralytics YOLO
  Vision-Language Model   OpenAI CLIP
  Embeddings              Sentence Transformers
  Retrieval               Text-based RAG
  Generative AI           Google Gemini
  Data Processing         Pandas / NumPy
  Deep Learning           PyTorch
  Model Repository        Hugging Face Hub

------------------------------------------------------------------------

# ⚙️ Installation

## 1. Clone the Repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd warehouse_vision_project
```

## 2. Create Virtual Environment

Windows:

``` powershell
python -m venv .venv
.venv\Scripts ctivate
```

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

``` powershell
python -m pip install -r requirements.txt
```

If necessary:

``` powershell
python -m pip install opencv-python ultralytics
```

------------------------------------------------------------------------

# 🔑 Gemini API Configuration

The recommendation stage requires a Gemini API key.

### Windows PowerShell

``` powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Verify:

``` powershell
python -c "import os; print('GEMINI KEY SET:', bool(os.getenv('GEMINI_API_KEY')))"
```

Expected:

``` text
GEMINI KEY SET: True
```

Never commit the API key to GitHub.

------------------------------------------------------------------------

# 🧪 Backend Testing

First test the backend import:

``` powershell
python -c "from pipeline import final_warehouse_pipeline; print('BACKEND OK')"
```

Then verify complete system initialization:

``` powershell
python -c "import pipeline; print('SYSTEM:', pipeline.SYSTEM is not None); print('ERROR:', getattr(pipeline, 'INITIALIZATION_ERROR', 'No initialization error'))"
```

Successful output:

``` text
SYSTEM: True
ERROR: No initialization error
```

------------------------------------------------------------------------

# ▶️ Run the Application

The current main UI filename contains a space.

Use:

``` powershell
streamlit run "Remmendation system_ui.py"
```

Then open:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 🔄 Complete Project Flow

``` text
┌──────────────────────────┐
│      User Uploads Image  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   OpenCV Preprocessing   │
│ Resize + Noise Reduction │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│      YOLO Detection      │
│ Bounding Box + Confidence│
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│    Product Crop          │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   CLIP Identification    │
│ Image ↔ Product Catalog  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Product Found      │
│ ID + Name + Category     │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Text RAG           │
│ Semantic Candidate Search│
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     RAG Candidates       │
│ Top Relevant Products    │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│       Gemini AI          │
│ Complementary Products   │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ Recommendation Validation│
│ Remove Invalid Products  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│    Final Recommendations │
└──────────────────────────┘
```

------------------------------------------------------------------------

# 📊 System Validation

The original project flow validates these stages:

``` text
✅ OpenCV preprocessing
✅ YOLO detection
✅ Product identification
✅ RAG retrieval
✅ Gemini recommendation
```

The final result contains:

``` text
status
yolo
product_detection
vision_top_predictions
rag_candidates
recommendations
```

------------------------------------------------------------------------

# ⚠️ Troubleshooting

### OpenCV Error

``` text
ModuleNotFoundError: No module named 'cv2'
```

Fix:

``` powershell
python -m pip install opencv-python
```

### Ultralytics Error

``` text
ModuleNotFoundError: No module named 'ultralytics'
```

Fix:

``` powershell
python -m pip install ultralytics
```

### Missing CSV

``` text
No such file or directory: cv_images.csv
```

Make sure the required CSV files are located beside `pipeline.py`.

Check:

``` powershell
dir *.csv
```

### Gemini Error

``` text
GEMINI_API_KEY environment variable is not set
```

Fix:

``` powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Restart Streamlit after setting the key.

### Streamlit File Error

If Streamlit reports that the file does not exist:

``` powershell
dir *.py
```

Then run the exact filename.

For the current project:

``` powershell
streamlit run "Remmendation system_ui.py"
```

------------------------------------------------------------------------

# 🔐 Security

Never commit:

``` text
API keys
Passwords
.env files
secrets.toml
Private credentials
```

Recommended `.gitignore`:

``` text
.venv/
__pycache__/
*.pyc
.env
.env.*
.streamlit/secrets.toml
```

------------------------------------------------------------------------

# 📌 Important Notes

The first run can take longer because pretrained models may need to be
downloaded.

The project uses pretrained models for:

-   YOLO
-   CLIP
-   Sentence Transformers

Hugging Face may display a warning about unauthenticated requests. This
is not necessarily an application failure; model downloads can still
work.

------------------------------------------------------------------------

# 🚀 Future Enhancements

Potential improvements include:

-   Real-time warehouse inventory availability
-   Warehouse location visualization
-   Product stock information
-   Recommendation explanations
-   Bounding-box visualization
-   Persistent recommendation history
-   User authentication
-   REST API deployment
-   Docker deployment
-   GPU acceleration
-   Automated testing
-   Production logging and monitoring

------------------------------------------------------------------------

# 👨‍💻 Project Summary

**Warehouse Vision-Based Pick Recommendation System**

``` text
Computer Vision
       +
Product Catalog
       +
Semantic Retrieval
       +
Generative AI
       =
AI-Based Warehouse Product Recommendation
```

### End-to-End Architecture

**Image → OpenCV → YOLO → CLIP → Text RAG → Gemini → Validation →
Recommendations**

------------------------------------------------------------------------

## 📜 License

Add the appropriate license before publishing the project.

If third-party datasets or pretrained models are redistributed, review
their individual licensing and usage requirements.
