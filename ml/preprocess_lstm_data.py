"""
preprocess_lstm_data.py

Prepares the ETIS text/email subset for LSTM training.

Scope decision (see CLAUDE.md "LSTM scope"): only content_type in
{'text', 'email'} is used here. URL rows are handled by the rule-based
and blacklist engines instead -- including them would let 84K+ mostly-
English URL rows numerically drown out the Arabic signal this model is
actually meant to learn.

Reuses the existing `split` column (stratified 70/15/15, seed 42) from
Phase 1 rather than re-splitting the subset independently, so the LSTM's
evaluation methodology stays consistent with the rest of the project.

Run from the project root:
    python ml/preprocess_lstm_data.py

Outputs (written to ml/data/):
    tokenizer.pkl           - fitted Keras Tokenizer (fit on train split only)
    X_train.npy / X_val.npy / X_test.npy   - padded, tokenized sequences
    y_train.npy / y_val.npy / y_test.npy   - labels (1 = phishing, 0 = legitimate)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Config -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data"

CONTENT_TYPES = ("text", "email")  # URLs excluded -- see module docstring
MAX_LEN = 200            # covers the 95th percentile of word length intact
VOCAB_SIZE = 5_000        # revised down from initial 20k-30k plan after inspecting the
                           # real frequency distribution: 9,913 of 16,737 training-vocab
                           # words (59%) appear exactly once, carrying no learnable signal
                           # for a 1,412-row training set. 5,000 covers essentially all
                           # words appearing 2+ times; the long singleton tail falls to <OOV>.
OOV_TOKEN = "<OOV>"


def normalize_arabic_text(text: str) -> str:
    """Dediacritize and normalize Arabic letter-form variants.

    Deliberately narrow: no morphological disambiguation. This dataset's
    vocabulary is phishing/URL-specific, not general Arabic prose, so
    heavier NLP would add cost without adding signal. camel-tools'
    normalize/dediac functions pass non-Arabic characters through
    unchanged, so this is safe to run on English and mixed-language rows
    too without a language check first.
    """
    if not isinstance(text, str):
        return ""
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text


def load_subset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    subset = df[df["content_type"].isin(CONTENT_TYPES)].copy()
    if subset["split"].isna().any():
        raise ValueError(
            "Found rows with an empty split column in the text/email subset. "
            "The stratified splitter should have assigned train/val/test to "
            "every row -- check data/scripts before proceeding."
        )
    return subset


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subset = load_subset()
    print(f"Loaded {len(subset)} text/email rows.")
    print(subset["split"].value_counts())

    subset["clean_text"] = subset["text"].apply(normalize_arabic_text)

    train_df = subset[subset["split"] == "train"]
    val_df = subset[subset["split"] == "val"]
    test_df = subset[subset["split"] == "test"]

    # Fit the tokenizer on the training split ONLY. Fitting on the full
    # subset would leak validation/test vocabulary into the token index,
    # which would make val/test metrics optimistic and non-representative
    # of real-world performance on unseen phishing language.
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(train_df["clean_text"])
    print(f"Tokenizer vocabulary size (train split only): {len(tokenizer.word_index)}")

    def to_padded(frame: pd.DataFrame) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(frame["clean_text"])
        return pad_sequences(
            sequences, maxlen=MAX_LEN, padding="post", truncating="post"
        )

    X_train = to_padded(train_df)
    X_val = to_padded(val_df)
    X_test = to_padded(test_df)

    label_map = {"legitimate": 0, "phishing": 1}
    y_train = train_df["label"].map(label_map).to_numpy()
    y_val = val_df["label"].map(label_map).to_numpy()
    y_test = test_df["label"].map(label_map).to_numpy()

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    with open(OUTPUT_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("\nShapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")
    print(f"\nSaved tokenizer and arrays to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
