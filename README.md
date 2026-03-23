🧠 MindTrack AI – Hybrid Mental Health Tracking System

MindTrack AI is a hybrid artificial intelligence system that combines deep learning and large language models to analyze and track mental well-being.

This project was developed as part of a Programming Languages course and demonstrates multiple neural network architectures along with real-world AI integration.

🚀 Features
🔹 1. Hybrid AI System
PyTorch Model → Classifies users as:
Risky
Balanced
Ideal
GPT Integration → Provides:
Personalized analysis
Actionable recommendations
Weekly reports
🔹 2. Model Comparison Module

Includes training and evaluation of 5 different architectures:

Baseline Neural Network
Deep Neural Network (BatchNorm + Dropout)
Residual Network (Skip Connections)
Attention Network (Feature Importance)
Ensemble Model (Weighted Combination)

📊 Model Evaluation
5-Fold Cross Validation
Accuracy, Precision, Recall, F1-score
Confusion Matrix & Radar Charts

🧠 User Tracking System
Daily data entry (sleep, stress, mood, etc.)
JSON-based local storage
Weekly analytics & visualization
GPT-powered insights

🗂️ Project Structure
mindtrack-ai/
│
├── model_comparison.py      # 5 model architecture comparison
├── main.py                  # Hybrid system (PyTorch + GPT)
│
├── data/
│   ├── data.json
│   └── model_checkpoint.pth
│
├── assets/
│   └── (graphs / dataset not included)
│
├── .env.example
├── requirements.txt
└── README.md

⚙️ Installation
git clone https://github.com/yourusername/mindtrack-ai.git
cd mindtrack-ai

pip install -r requirements.txt
🔑 Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
▶️ Usage
Train the model
python main.py

Then select:

6) Train Model
Run hybrid system
python main.py
Run model comparison
python model_comparison.py
📊 Dataset

This project uses:

Sleep Health and Lifestyle Dataset

⚠️ Dataset is not included due to licensing.

You can download it from:
https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset

🧠 Technologies Used
PyTorch
NumPy / Pandas
Scikit-learn
Matplotlib / Seaborn
OpenAI API
📈 Example Outputs
Model comparison graphs
Confusion matrices
Weekly mental health trends
GPT-generated reports
⚠️ Disclaimer

This project is for educational purposes only and is not a medical diagnosis tool.

👨‍💻 Author

Berke Genç
Software Engineering Student
