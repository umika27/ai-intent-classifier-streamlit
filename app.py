import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# --------------- Load model ---------------- #

model_path = "intent_deep_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
id2label = model.config.id2label


# --------------- Style Response Templates ---------------- #

def apply_style(intent, base_response, style):

    if style == "Overconfident Genius":
        return f"Obviously, it’s pretty clear this is **{intent}**. {base_response} Anyone could see that."

    if style == "Nervous Intern":
        return f"Um… I think this is **{intent}**. I-I might be wrong but {base_response}"

    if style == "Sarcastic Reviewer":
        return f"Oh wow, totally shocking. It’s **{intent}**. Anyway— {base_response}"

    if style == "Calm Professor":
        return f"Let’s go through this slowly. The intent here is **{intent}**. {base_response}"
    if style == "Shakespeare":
        return f"Verily, the intent revealeth itself as **{intent}**.{base_response}"
    if style == "Toddler":
        return f"Hehe, this is **{intent}**!{base_response}"
    return f"The detected intent is **{intent}**. {base_response}"


# --------------- Base responses ---------------- #

def generate_base_response(intent):
    if intent == "Academic":
        return "This question relates to educational or conceptual learning topics."
    if intent == "Technical":
        return "This query is related to programming, computing, or technology."
    if intent == "Entertainment":
        return "This is about movies, songs, humor, or fun activities."
    if intent == "Personal":
        return "This expresses personal emotions, feelings, or life situations."
    if intent == "General":
        return "This is a broad general knowledge or everyday query."
    return "I’m not fully sure, but this is my best interpretation."


# --------------- Intent classifier ---------------- #

def classify_with_style(text, style):

    if text.strip() == "":
        return "Please enter some text 🙂"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()

    intent = id2label[label_id]
    base_response = generate_base_response(intent)

    return apply_style(intent, base_response, style)


# --------------- STREAMLIT UI ---------------- #

st.title("🎯 Intent Detection + Style-based Response AI")
st.write("Deep Learning model fine-tuned on DistilBERT")

user_input = st.text_area("Enter your text here:")

style = st.selectbox(
    "Choose a response style:",
    [
        "Neutral",
        "Overconfident Genius",
        "Nervous Intern",
        "Sarcastic Reviewer",
        "Calm Professor",
        "Shakespeare",
        "Toddler"
    ]
)

if st.button("Generate Response"):
    output = classify_with_style(user_input, style)
    st.markdown(f"### 🧠 Response:")
    st.markdown(output)
