import numpy as np
from sklearn.model_selection import train_test_split

# Split data
def split_data(df):
    X = df["review"]
    y = df["sentiment"].map({"negative": 0, "positive": 1})
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Word2Vec
def tokenize(texts):
    return [text.lower().split() for text in texts]

def average_vectors(tokenized, model):
    return np.array([
        np.mean([model.wv[w] for w in tokens if w in model.wv], axis=0)
        if any(w in model.wv for w in tokens)
        else np.zeros(model.vector_size)
        for tokens in tokenized
    ])
