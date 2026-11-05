import numpy as np


def _sigmoid(z):
    """The logistic sigmoid function with numerical stability clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def lstm_forward(X, Wx, Wh, b, h0, c0):
    """Forward pass of an LSTM over sequence X."""
    T, d = X.shape
    h = h0.shape[0]
    
    H = np.zeros((T, h))
    C = np.zeros((T, h))
    cache = []
    
    h_prev = h0
    c_prev = c0
    
    for t in range(T):
        x = X[t]
        z = Wx @ x + Wh @ h_prev + b
        
        zi = z[0:h]
        zf = z[h:2*h]
        zo = z[2*h:3*h]
        zg = z[3*h:4*h]
        
        i = _sigmoid(zi)
        f = _sigmoid(zf)
        o = _sigmoid(zo)
        g = np.tanh(zg)
        
        c = f * c_prev + i * g
        tanh_c = np.tanh(c)
        h_val = o * tanh_c
        
        H[t] = h_val
        C[t] = c
        cache.append((x, h_prev, c_prev, i, f, o, g, c))
        
        h_prev = h_val
        c_prev = c
        
    return H, C, cache


def lstm_backward(dH, cache, Wx, Wh):
    """Backpropagation Through Time (BPTT) for an LSTM."""
    T, h = dH.shape
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros((4 * h,))
    
    dh_next = np.zeros((h,))
    dc_next = np.zeros((h,))
    
    for t in reversed(range(T)):
        x, h_prev, c_prev, i, f, o, g, c = cache[t]
        
        dh_total = dH[t] + dh_next
        
        tanh_c = np.tanh(c)
        do = dh_total * tanh_c
        dtanh_c = dh_total * o
        dc_total = dc_next + dtanh_c * (1.0 - tanh_c ** 2)
        
        df = dc_total * c_prev
        di = dc_total * g
        dg = dc_total * i
        
        # Derivatives of activations
        dzi = di * i * (1.0 - i)
        dzf = df * f * (1.0 - f)
        dzo = do * o * (1.0 - o)
        dzg = dg * (1.0 - g ** 2)
        
        dz = np.concatenate([dzi, dzf, dzo, dzg])
        
        dWx += np.outer(dz, x)
        dWh += np.outer(dz, h_prev)
        db += dz
        
        dh_next = Wh.T @ dz
        dc_next = dc_total * f
        
    dh0 = dh_next
    dc0 = dc_next
    
    return dWx, dWh, db, dh0, dc0


def sequence_loss(X, Wx, Wh, b, targets):
    """Mean squared error loss between sequence predictions and targets, and its gradient."""
    h = Wh.shape[1]
    h0 = np.zeros((h,))
    c0 = np.zeros((h,))
    
    H, C, cache = lstm_forward(X, Wx, Wh, b, h0, c0)
    
    loss = np.mean((H - targets) ** 2)
    T, h_dim = H.shape
    dH = 2.0 * (H - targets) / (T * h_dim)
    
    return loss, dH