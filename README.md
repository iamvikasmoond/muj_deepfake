# Deepfake Detection using Deep Learning

A deep learning-based web application for detecting deepfake videos using a **ResNext50 + LSTM** architecture. Trained on the Celeb-DF and FaceForensics++ datasets, achieving up to **97% validation accuracy**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Trained Models](#trained-models)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training](#training)
- [Running the Web Application](#running-the-web-application)
- [How It Works](#how-it-works)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

Deepfakes are AI-generated videos where a person's face is replaced with someone else's. This project detects such manipulations by analyzing temporal patterns across video frames using a hybrid CNN-LSTM model.

The system:
- Accepts an uploaded video via a web interface
- Extracts and crops faces from frames
- Passes them through a ResNext50 + LSTM model
- Returns a REAL/FAKE prediction with a confidence score

---

## Architecture

```
Video Input
    ↓
Frame Extraction (20/40/60/80 frames)
    ↓
Face Detection & Cropping (face_recognition + dlib)
    ↓
ResNext50 (pretrained on ImageNet) — Feature Extraction
    ↓
LSTM — Temporal Pattern Analysis
    ↓
Softmax → REAL / FAKE + Confidence Score
```

**ResNext50** extracts spatial features from each frame. The **LSTM** layer learns temporal inconsistencies across the sequence of frames — something that distinguishes real faces from synthetically generated ones.

Transfer learning is used: ResNext50 weights are frozen (pretrained on ImageNet), and only the LSTM and linear layers are trained.

---

## Datasets

| Dataset | Type | Videos Used |
|---------|------|-------------|
| Celeb-DF v2 | Fake + Real | 580 Fake, 588 Real |
| FaceForensics++ | Fake + Real | 996 Fake, 993 Real |
| **Total** | | **3,157 videos** |

All videos are preprocessed — face-only crops extracted from raw videos.

### Download Preprocessed Data

| Dataset | Link |
|---------|------|
| Celeb-DF Fake | [Google Drive](https://drive.google.com/drive/folders/1SxCb_Wr7N4Wsc-uvjUl0i-6PpwYmwN65) |
| Celeb-DF Real | [Google Drive](https://drive.google.com/drive/folders/1g97v9JoD3pCKA2TxHe8ZLRe4buX2siCQ) |
| FaceForensics++ | [Google Drive](https://drive.google.com/drive/folders/1VIIWRLs6VBXRYKODgeOU7i6votLPPxT0) |

Labels: `Model Creation/labels/Gobal_metadata.csv`

---

## Trained Models

All models are stored in `Django Application/models/`. The web app automatically selects the best model based on the sequence length entered in the UI.

| Model File | Sequence Length | Accuracy | Dataset |
|------------|----------------|----------|---------|
| `model_84_acc_10_frames_final_data.pt` | 10 frames | 84% | Celeb-DF |
| `model_87_acc_20_frames_final_data.pt` | 20 frames | 87% | Celeb-DF |
| `model_89_acc_40_frames_final_data.pt` | 40 frames | 89% | Celeb-DF + FF++ |
| `model_90_acc_60_frames_final_data.pt` | 60 frames | 90% | Celeb-DF + FF++ |
| `model_97_acc_80_frames_FF_data.pt` | 80 frames | 97% | FaceForensics++ |

**Recommendation:** Use 20 frames for speed, 60-80 frames for accuracy.

---

## Project Structure

```
Deepfake_detection_using_deep_learning/
├── Model Creation/
│   ├── dataset/
│   │   ├── Fake/               ← Preprocessed fake videos
│   │   └── Real/               ← Preprocessed real videos
│   ├── labels/
│   │   └── Gobal_metadata.csv  ← Video labels
│   ├── Helpers/
│   │   ├── Create_csv_from_glob.ipynb
│   │   └── ...
│   ├── Model_and_train_csv.ipynb  ← Main training notebook
│   └── Predict.ipynb              ← Standalone prediction
│
├── Django Application/
│   ├── models/                 ← Trained .pt model files
│   ├── ml_app/
│   │   ├── views.py            ← Core prediction logic
│   │   ├── templates/
│   │   │   ├── index.html      ← Upload page
│   │   │   └── predict.html    ← Results page
│   │   └── urls.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── nav-bar.html
│   │   └── footer.html
│   ├── project_settings/
│   │   └── settings.py
│   ├── uploaded_videos/        ← Uploaded videos stored here
│   ├── uploaded_images/        ← Frame crops stored here
│   └── manage.py
│
└── Documentation/
```

---

## Setup & Installation

### Requirements

- Python 3.11
- Mac (Apple Silicon M1/M2/M3/M4) or Linux
- 8GB+ RAM recommended

### 1. Clone the repository

```bash
git clone https://github.com/iamvikasmoond/muj_deepfake.git
cd muj_deepfake
```

### 2. Create virtual environment

```bash
python3.11 -m venv deepfake-env
source deepfake-env/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install django==3.2 opencv-python-headless face-recognition \
    numpy pandas scikit-learn matplotlib seaborn pillow ipykernel
```

> **Note for Apple Silicon (M1/M2/M3/M4):** PyTorch MPS backend is included by default. No CUDA needed. Training runs on the GPU automatically.

> **dlib installation:** If `face_recognition` fails, run `brew install cmake dlib` first.

### 4. Verify MPS (Apple Silicon only)

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

---

## Training

### 1. Download and place dataset

Place preprocessed videos in:
```
Model Creation/dataset/Fake/   ← fake videos
Model Creation/dataset/Real/   ← real videos
```

### 2. Generate CSV files

Run `Model Creation/Helpers/Create_csv_from_glob.ipynb`

This creates:
- `dataset/real_youtube.csv`
- `dataset/fake_celeb.csv`

### 3. Train the model

Open `Model Creation/Model_and_train_csv.ipynb` in VS Code with the `deepfake-env` kernel selected.

**Key hyperparameters:**

```python
sequence_length = 20    # frames per video (10/20/40/60/80)
batch_size      = 4     # lower if MPS memory errors occur
num_epochs      = 20
lr              = 1e-5  # Adam optimizer
```

**MPS patch (Apple Silicon — already applied in notebook):**

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

### 4. Save trained model

The notebook saves checkpoints automatically after each epoch:
```
Model Creation/checkpoint.pt
```

Copy to Django models folder with proper naming:
```bash
cp "Model Creation/checkpoint.pt" \
   "Django Application/models/model_{acc}_acc_{seq}_frames_{dataset}.pt"
```

---

## Running the Web Application

### 1. Activate environment

```bash
source ~/deepfake-env/bin/activate
cd "Django Application"
```

### 2. Start server

```bash
python manage.py runserver
```

### 3. Open in browser

```
http://127.0.0.1:8000
```

### 4. Upload a video

1. Click **Drop Media Here** or drag and drop a video
2. Select **Sequence Length** (10/20/40/60/80) — higher = more accurate but slower
3. Click **Analyze Now**
4. View results — confidence score, frame splits, face crops, and video info

The app automatically selects the best trained model for the chosen sequence length.

---

## How It Works

### Model Selection

The Django app scans `models/` folder and selects the highest accuracy model matching the sequence length:

```python
# Filename format: model_{acc}_acc_{seq}_frames_{dataset}.pt
# Example: model_87_acc_20_frames_final_data.pt
# Entering sequence_length=20 → picks this model automatically
```

### Prediction Pipeline

1. Video uploaded → saved to `uploaded_videos/`
2. Frames extracted using OpenCV
3. Faces detected using `face_recognition` library
4. Face crops saved to `uploaded_images/`
5. Frames passed through ResNext50 → feature vectors
6. Feature sequence passed through LSTM → classification
7. Softmax → confidence score + REAL/FAKE label
8. Results rendered on prediction page

---

## Results

| Sequence Length | Validation Accuracy | False Positive Rate | False Negative Rate |
|----------------|--------------------|--------------------|---------------------|
| 10 frames | 84% | ~16% | ~16% |
| 20 frames | 87% | ~13% | ~13% |
| 40 frames | 89% | ~11% | ~11% |
| 60 frames | 90% | ~10% | ~10% |
| 80 frames | 97% | ~3% | ~3% |

### Confusion Matrix (20-frame model, Celeb-DF)

```
              Predicted
              FAKE    REAL
Actual FAKE   100     9
       REAL   22      100
```

Accuracy: **86.75%** | Balanced errors — no class bias

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch (MPS / CUDA / CPU) |
| CNN Backbone | ResNext50_32x4d (pretrained ImageNet) |
| Sequence Model | LSTM |
| Face Detection | face_recognition + dlib |
| Video Processing | OpenCV |
| Web Framework | Django 3.2 |
| Frontend | HTML5, CSS3, JavaScript |
| Dataset | Celeb-DF v2, FaceForensics++ |
| Training Hardware | Apple Mac mini M4 (MPS backend) |

---

## Limitations

- High confidence scores on in-distribution data (same dataset as training)
- May struggle with low quality, partially occluded, or heavily compressed videos
- Not tested on newer deepfake generation methods (e.g., diffusion-based fakes)
- Real-world generalization requires training on more diverse datasets like DFDC

---

## Course Information

**Course Code:** AI4270 — Major Project  
**Institution:** Manipal University Jaipur  
**Specialization:** B.Tech Computer Science & Engineering (AI/ML)  
**Year:** 2025-2026

---

## References

1. Li, Y. et al. "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics." CVPR 2020.
2. Rössler, A. et al. "FaceForensics++: Learning to Detect Manipulated Facial Images." ICCV 2019.
3. Xie, S. et al. "Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)." CVPR 2017.
