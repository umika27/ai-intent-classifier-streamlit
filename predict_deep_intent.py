from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
# -------- Style Response Templates -------- #

def apply_style(intent, base_response, style):

    if style == "Overconfident Genius":
        return f"Obviously, it’s pretty clear this is **{intent}**. {base_response} Anyone could see that."

    if style == "Nervous Intern":
        return f"Um… I think this is **{intent}**. I-I might be wrong but {base_response}"

    if style == "Sarcastic Reviewer":
        return f"Oh wow, shocking. Totally unexpected. It’s **{intent}**. Anyway— {base_response}"

    if style == "Calm Professor":
        return f"Let’s understand this slowly. The intent here is **{intent}**. {base_response}"

    # default neutral
    return f"Detected intent: **{intent}**. {base_response}"


model_path = "intent_deep_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

id2label = model.config.id2label
def generate_base_response(intent):
    if intent == "Academic":
        return "This question relates to learning, concepts, or educational topics."
    if intent == "Technical":
        return "This query is related to technology, programming, or computing."
    if intent == "Entertainment":
        return "This is about fun content like movies, songs, or humor."
    if intent == "Personal":
        return "This reflects emotions, feelings, or personal life situations."
    if intent == "General":
        return "This seems like a general everyday question or casual statement."
    return "I’m not fully sure, but this is my best interpretation."

def classify_with_style(text, style="Neutral"):
    text = text.strip()
    if text == "":
        return "No input provided."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()

    intent = id2label[label_id]
    base_response = generate_base_response(intent)

    return apply_style(intent, base_response, style)

styles = [
    "Neutral",
    "Overconfident Genius",
    "Nervous Intern",
    "Sarcastic Reviewer",
    "Calm Professor",
    "Shakespeare",
    "Toddler"
]

print("Available styles:")
for s in styles:
    print("-", s)

while True:
    q = input("\nEnter text: ")
    st = input("Choose style: ")

    if st not in styles:
        st = "Neutral"

    print("\nResponse:")
    print(classify_with_style(q, st))

