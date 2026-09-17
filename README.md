# 🎭 Emotion Detection from Uploaded Images

## 📌 Project Overview
This project focuses on detecting **human emotions** from images using **Convolutional Neural Networks (CNNs)** integrated within a **Streamlit web application**.  
Users can upload an image, and the system detects the **face**, extracts **facial features**, and classifies the **emotion** (e.g., happy, sad, angry, surprised, neutral, etc.).

---

## 🎯 Objective
To develop a comprehensive **end-to-end system** that:
- Accepts user-uploaded images via Streamlit.
- Detects and extracts faces.
- Classifies the emotion using a trained CNN model.
- Provides accurate and real-time predictions through a simple, user-friendly interface.

---

## 🧠 Key Components

### 1. User Interface (UI)
- Built with **Streamlit** for simplicity and interactivity.  
- Restricts upload to **image files only** (formats like `.jpg`, `.jpeg`, `.png`).  
- Provides clear error messages for invalid files or large sizes.

### 2. Facial Detection
- Utilizes **OpenCV** and **Mediapipe/Dlib** for face detection.
- Resizes and crops detected face regions before feeding them into the CNN.
- Combines pre-trained and custom detection techniques for improved accuracy.

### 3. Facial Feature Extraction
- Extracts **key landmarks** (eyes, mouth, eyebrows, etc.) using **Dlib** or **Mediapipe**.
- Landmark precision improves emotion classification by focusing on critical features.

### 4. Emotion Classification
- CNN trained using **FER-2013 dataset** (available on Kaggle or via `torchvision.datasets.FER2013`).
- Tested multiple architectures (e.g., VGG-like CNN, ResNet, custom CNN).
- Evaluated performance using **accuracy**, **precision**, **recall**, and **F1-score**.

### 5. Optimization and Performance
- Real-time inference with optimized CNN weights.
- Preprocessing pipelines ensure smooth handling of uploaded images.

---

## ⚙️ Tools & Technologies
| Category | Tools Used |
|-----------|-------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Web Framework | Streamlit |
| Libraries | OpenCV, Mediapipe/Dlib, NumPy, Pandas, Matplotlib |
| Dataset | FER-2013 |
| Environment | VS Code / Jupyter Notebook |

---

## 📊 Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

---

## 🧩 Results
- The model achieved **high accuracy** in detecting emotions such as *Happy*, *Sad*, *Angry*, and *Neutral*.
- The Streamlit app provides real-time predictions and smooth user interaction.

---

## 🧱 Project Structure
│
├── app.py # Streamlit application
├── model/
│ ├── emotion_model.pth # Trained CNN model
│ ├── model_training.ipynb # Model training notebook
│
├── utils/
│ ├── face_detection.py # Face detection script
│ ├── preprocessing.py # Image preprocessing functions
│
├── dataset/ # FER-2013 or sample data (optional)
├── requirements.txt # Required dependencies
├── README.md # Project documentation
└── report/ # Project report and ethical analysis
