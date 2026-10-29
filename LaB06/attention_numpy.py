import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    attention_weights = softmax(scores)
    context = attention_weights @ V
    return scores, attention_weights, context
