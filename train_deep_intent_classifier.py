import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# -------- Load dataset -------- #
df = pd.read_csv("dataset.csv")

texts = df["text"].tolist()
labels = df["label"].astype("category")
label2id = {c: i for i, c in enumerate(labels.cat.categories)}
id2label = {i: c for c, i in label2id.items()}
y = labels.cat.codes.tolist()

def classify(text):
    text = text.strip()
    if text == "":
        return "No input provided"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs).item()
    return id2label[label_id]

# -------- Train-test split -------- #
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, y, test_size=0.2, random_state=42
)

# -------- Tokenizer -------- #
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_enc = tokenizer(train_texts, truncation=True, padding=True)
test_enc = tokenizer(test_texts, truncation=True, padding=True)

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_enc, train_labels)
test_dataset = IntentDataset(test_enc, test_labels)

# -------- Model -------- #
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# -------- Training arguments -------- #
training_args = TrainingArguments(
    output_dir="./intent_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)



# -------- Trainer -------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# -------- Save model -------- #
model.save_pretrained("intent_deep_model")
tokenizer.save_pretrained("intent_deep_model")

print("Deep learning model saved in folder 'intent_deep_model'")
