# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x, alpha):
    return np.where(x >= 0, x, x * alpha)

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x*tanh(softplus(x))

def softplus(x):
    return np.log(1.0 + np.exp(x))

# 参考
# https://qiita.com/Hatomugi/items/d00c1a7df07e0e3925a8#%E3%83%92%E3%83%B3%E3%82%B8%E6%90%8D%E5%A4%B1-hinge-loss
# https://atmarkit.itmedia.co.jp/ait/articles/2108/04/news031.html
# https://aizine.ai/glossary-loss-function/#toc4

# 二乗誤差
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 平均絶対誤差(MAE)
def mean_absolute_error(y, t):
    return np.sum(np.abs(y-t))/len(y)

# 平均二乗誤差(MSE)
def mean_absolute_error(y, t):
    return np.sum((y-t)**2)/len(y)



def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
