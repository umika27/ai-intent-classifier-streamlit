# 🎯 AI Intent Detection with Style-Based Responses

This project is an AI-based **intent classifier** that identifies the intent of user input and generates responses in different **personality styles**.

✔ Fine-tuned **DistilBERT deep learning model**  
✔ Detects intents:
- Academic
- Technical
- Entertainment
- Personal
- General

✔ Responds in multiple styles:
- Neutral
- Overconfident Genius 😎
- Nervous Intern 😬
- Sarcastic Reviewer 🙄
- Calm Professor 🧑‍🏫
- Shakespeare ✒️
- Toddler 🍼

✔ Web interface built using **Streamlit**

---

## 🧠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Streamlit

---

## 📂 Project Structure
```bash
AIMLProject/
│
├── dataset.csv
├── train_deep_intent_classifier.py
├── predict_deep_intent.py
├── app.py
├── intent_deep_model/ # saved model & tokenizer folder after training
└── README.md
```
---

## 🚀 How to Run the Project

### 1️⃣ Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Mac / Linux
venv\Scripts\activate      # Windows
```
### 2️⃣ Install requirements
```bash
pip install torch transformers scikit-learn pandas streamlit
```
If torch fails on Mac Silicon:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
``` 
### 3️⃣ Train the model
```bash
python train_deep_intent_classifier.py
```
This:
- trains DistilBERT
- saves model into /intent_deep_model
## 4️⃣ Run Streamlit Web App
```bash
streamlit run app.py
```
Then open browser link (usually http://localhost:8501)

---

## 🧪 How it Works

- User enters a text query
- Deep learning model predicts intent class
- System selects a base response
- Chosen style personality rewrites tone
- Final message displayed in UI

--- 

## 🎓 Example Outputs

- Input: Tell me a joke
- Style: Sarcastic Reviewer
- ✔ Entertainment intent detected
- ✔ Sarcastic response generated

--- 

## 📌 Future Improvements

- Add emotion detection (happy / sad / angry)
- Add speech-to-text and text-to-speech
- Deploy to HuggingFace / Streamlit Cloud
- Add multilingual support

---

## 👩‍💻 Author

Project by **Umika Sood**

---
