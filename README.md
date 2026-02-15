# Sign Language Recognition System

This project implements a real-time Sign Language Recognition system using:
- MediaPipe for hand landmark detection  
- ML model for gesture prediction with algorithm RFC
- Gradio for interactive web interface  

The system captures live webcam input, extracts hand landmarks, predicts characters, and builds words in real-time.

---

## How to Run the Project locally

#### 1. Clone the Repository

```bash
git clone https://github.com/adarshkatare6/Sign-Language-Recognizer-AI
cd Sign-Language-Recognizer-AI
```

#### 2. Create a Virtual Environment

##### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

##### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

#### 4. Run the Application

```bash
python app.py
```

After running, open your browser and go to:

```
http://127.0.0.1:7860
```

Allow webcam access when prompted.

---

## Requirements

- Python 3.9 – 3.11 recommended  
- Webcam access  

To check Python version:

```bash
python --version
```

---

## Future Improvements

- Add sentence formation
- Improve prediction stability
- Deploy using Docker or Hugging Face
- Add multilingual gesture support

---
