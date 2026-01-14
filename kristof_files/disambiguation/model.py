import pandas as pd
import fasttext
import re
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Paths
TRAIN_CSV = "train.csv"
TEST_CSV = "eval.csv"

TRAIN_TEXT_COL = "text"
TRAIN_LABEL_COL = "lang"

TEST_TEXT_COL = "text"
TEST_LABEL_COL = "lang"

MODEL_NAME = "fin-fit-disambiguation-model.bin"

# Config
MIN_TEXT_LEN = 20
CONF_THRESHOLD = 0.0

# FastText hyperparameters
EPOCHS = 25
LR = 0.5
WORD_NGRAMS = 3
DIM = 100
BUCKET = 200_000
MIN_COUNT = 1
LOSS = "softmax"

# Label normalization
LABEL_MAP = {
    "finnish": "fin",
    "fin": "fin",
    "meänkieli": "fit",
    "meankieli": "fit",
    "fit": "fit",
}

# Text cleaning
def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "0", text)
    return text

# Labels normalized BEFORE training
def load_csv(path, text_col, label_col):
    df = pd.read_csv(path)

    df = df[[text_col, label_col]]
    df.columns = ["text", "lang"]

    df.dropna(inplace=True)

    df["lang"] = (
        df["lang"]
        .str.lower()
        .map(LABEL_MAP)
    )

    # Drop unmapped labels
    df = df[df["lang"].notna()]

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() >= MIN_TEXT_LEN]

    return df.reset_index(drop=True)

# FastText formatting
def to_fasttext(df):
    return "__label__" + df["lang"] + " " + df["text"]

# Training
print("Loading training data...")
train_df = load_csv(
    TRAIN_CSV,
    TRAIN_TEXT_COL,
    TRAIN_LABEL_COL
)

print(f"Training samples: {len(train_df)}")
print("Training labels:", sorted(train_df["lang"].unique()))

Path("data").mkdir(exist_ok=True)
TRAIN_TXT = "data/train.txt"

to_fasttext(train_df).to_csv(
    TRAIN_TXT,
    index=False,
    header=False
)

print("\nTraining fastText model...")
model = fasttext.train_supervised(
    input=TRAIN_TXT,
    epoch=EPOCHS,
    lr=LR,
    wordNgrams=WORD_NGRAMS,
    dim=DIM,
    bucket=BUCKET,
    minCount=MIN_COUNT,
    loss=LOSS
)

model.save_model(MODEL_NAME)
print(f"Model saved to file {MODEL_NAME}")

# Prediction function
def predict_lang(model, text, conf_threshold=0.0):
    labels, probs = model.predict(text, k=1)

    label = labels[0].replace("__label__", "")  # 🔴 CHANGED
    prob = float(probs[0])

    if prob < conf_threshold:
        return "unknown", prob

    return label, prob

# Evaluation
print("\nLoading test data...")
test_df = load_csv(
    TEST_CSV,
    TEST_TEXT_COL,
    TEST_LABEL_COL
)

print(f"Test samples: {len(test_df)}")
print("Test labels:", sorted(test_df["lang"].unique()))

preds = []
probs = []

for text in test_df["text"]:
    pred, prob = predict_lang(model, text, CONF_THRESHOLD)  # 🔴 CHANGED
    preds.append(pred)
    probs.append(prob)

test_df["pred"] = preds
test_df["prob"] = probs

# Metrics
eval_df = test_df[test_df["pred"] != "unknown"]

print("\nAccuracy (excluding rejected samples):")
print(accuracy_score(eval_df["lang"], eval_df["pred"]))

print("\nClassification report:")
print(
    classification_report(
        eval_df["lang"],
        eval_df["pred"],
        digits=4
    )
)

labels = sorted(set(test_df["lang"]) | set(test_df["pred"]))
cm = confusion_matrix(
    test_df["lang"],
    test_df["pred"],
    labels=labels
)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("\nConfusion matrix:")
print(cm_df)

# Errors
errors = test_df[test_df["lang"] != test_df["pred"]]
errors.to_csv("errors.csv", index=False)

print(f"\nSaved {len(errors)} misclassified examples to errors.csv")

# Print missclassifications
#print("\nMisclassified texts:\n")
#
#for _, row in errors.iterrows():
#    print(
#        f"TRUE: {row['lang']} | "
#        f"PRED: {row['pred']} | "
#        f"PROB: {row['prob']:.3f}\n"
#        f"{row['text']}\n"
#        f"{'-'*80}"
#    )

# Clean label output (no __label__)
examples = ["Stockholmissa on", "Sinä olet", "Jag är inte", "Blablabla"]
for text in examples:
    print(text, predict_lang(model, text))