# DeepFake_video_detection-Using-hybrid-model

A Flask-based web app that detects deepfake videos using **spatial-temporal hybrid models** (EfficientNet/MobileNet + LSTM/GRU).

## 🔥 Key Features
- **Hybrid Architecture**: Combines CNNs (EfficientNet, MobileNetV3, ResNeXt) + RNNs (LSTM/GRU) for frame + temporal analysis.
- **Multiple Models**: Choose between 6 optimized models (balance speed/accuracy).
- **Real-Time Ready**: MobileNetV3+GRU runs at **<1ms inference** on GPUs.

## 🛠️ Requirements
python=>3.10X
torch==2.3.0 
torchvision==0.18.0
opencv-python-headless==4.9.0.80
mtcnn==0.1.1 
numpy==1.26.4 
pandas==2.2.2 
scikit-learn==1.4.2 
tqdm==4.66.2 
matplotlib==3.8.3 
seaborn==0.13.2



**Models are hosted on Google Drive** (too large for GitHub LFS):  
📦 [Download Models ZIP](https://drive.google.com/drive/folders/1c39gN4YYrjMWzhi3Xf6ng-bn1tbs3vaz?usp=drive_link) 
- create a folder Named models inside app folder there store the downloded models with same names as in the drive.

🛠️ Installation
**1. Clone the Repository**

**2.Download the requirements mentioned above**

**3.Download the models from the Drive**

**4.Run the application**
   python app.py

Access the web interface at:
🌐 http://localhost:5000

DeepFake_video_detection-Using-hybrid-model/
├── models/
│   ├── efficientnet_gru.pt
│   ├── mobilenet_lstm.pt
│   └── ... (other .pt files)
├── app.py
└── README.md

**🎯 Usage**
**1.Upload a video (MP4/AVI/MOV).**

**2.Select a model (default: ResNeXt-LSTM for highest accuracy).**

**3.Get prediction:**

✅ REAL/FAKE result

📊 Confidence scores

Results:
Mobilenet+GRU:88.61%(Accuracy)
For CPU environment:
![image](https://github.com/user-attachments/assets/c37d0b10-c702-4434-80d0-5a90a471e479)
For kaggle GPU(T4) environment:
![image](https://github.com/user-attachments/assets/a0954a2a-622f-4a3e-8b2e-3fddf80ef8db)

Dataset: FaceForensics++, Celeb-DF
Libraries: PyTorch, Flask, OpenCV


Contact Details in profile




   


