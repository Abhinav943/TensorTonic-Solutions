import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    samples,features = X.shape
    W = np.zeros(features)
    b = 0

    for _ in range(steps):
        linear_model = np.dot(X,W) + b
        y_pred = _sigmoid(linear_model)

        dw = (1/samples)*(np.dot(X.T,(y_pred - y)))
        db = (1/samples)*(np.sum(y_pred - y))

        W -= lr*dw
        b -= lr*db

    return (W,b)
    
    pass