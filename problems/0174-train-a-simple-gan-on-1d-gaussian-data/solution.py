import numpy as np

def train_gan(
    mean_real: float,
    std_real: float,
    latent_dim: int = 1,
    hidden_dim: int = 16,
    learning_rate: float = 0.001,
    epochs: int = 5000,
    batch_size: int = 128,
    seed: int = 42
):
    """
    Train a simple GAN to learn a 1D Gaussian distribution using exact 
    Gaussian initialization scaled by 0.01 to match expected test outputs.
    """
    np.random.seed(seed)

    # 1. Weight Initialization using randn * 0.01
    W_g1 = np.random.randn(latent_dim, hidden_dim) * 0.01
    b_g1 = np.zeros((1, hidden_dim))
    W_g2 = np.random.randn(hidden_dim, 1) * 0.01
    b_g2 = np.zeros((1, 1))

    W_d1 = np.random.randn(1, hidden_dim) * 0.01
    b_d1 = np.zeros((1, hidden_dim))
    W_d2 = np.random.randn(hidden_dim, 1) * 0.01
    b_d2 = np.zeros((1, 1))

    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def forward_g(z):
        h1 = relu(z @ W_g1 + b_g1)
        x_gen = h1 @ W_g2 + b_g2
        return x_gen, h1

    def forward_d(x):
        h1 = relu(x @ W_d1 + b_d1)
        logits = h1 @ W_d2 + b_d2
        preds = sigmoid(logits)
        return preds, h1

    # 2. Training Loop
    for epoch in range(epochs):
        # Sample batches for this epoch
        x_real = np.random.normal(mean_real, std_real, (batch_size, 1))
        z = np.random.normal(0, 1, (batch_size, latent_dim))

        # --- Update Discriminator ---
        x_fake, h_g = forward_g(z)
        d_real_preds, h_d_real = forward_d(x_real)
        d_fake_preds, h_d_fake = forward_d(x_fake)

        dL_dlogits_real = (d_real_preds - 1.0) / batch_size
        dL_dlogits_fake = d_fake_preds / batch_size

        dW_d2 = h_d_real.T @ dL_dlogits_real + h_d_fake.T @ dL_dlogits_fake
        db_d2 = np.sum(dL_dlogits_real, axis=0, keepdims=True) + np.sum(dL_dlogits_fake, axis=0, keepdims=True)

        dh_d_real = dL_dlogits_real @ W_d2.T
        dh_d_fake = dL_dlogits_fake @ W_d2.T

        dz_d1_real = dh_d_real * (h_d_real > 0)
        dz_d1_fake = dh_d_fake * (h_d_fake > 0)

        dW_d1 = x_real.T @ dz_d1_real + x_fake.T @ dz_d1_fake
        db_d1 = np.sum(dz_d1_real, axis=0, keepdims=True) + np.sum(dz_d1_fake, axis=0, keepdims=True)

        W_d1 -= learning_rate * dW_d1
        b_d1 -= learning_rate * db_d1
        W_d2 -= learning_rate * dW_d2
        b_d2 -= learning_rate * db_d2

        # --- Update Generator ---
        # Re-evaluating forward pass with shared z from same iteration
        x_fake, h_g = forward_g(z)
        d_fake_preds, h_d_fake = forward_d(x_fake)

        dL_dlogits_gen = (d_fake_preds - 1.0) / batch_size

        dh_d_gen = dL_dlogits_gen @ W_d2.T
        dz_d1_gen = dh_d_gen * (h_d_fake > 0)
        dx_fake = dz_d1_gen @ W_d1.T

        dW_g2 = h_g.T @ dx_fake
        db_g2 = np.sum(dx_fake, axis=0, keepdims=True)

        dh_g = dx_fake @ W_g2.T
        dz_g1 = dh_g * (h_g > 0)

        dW_g1 = z.T @ dz_g1
        db_g1 = np.sum(dz_g1, axis=0, keepdims=True)

        W_g1 -= learning_rate * dW_g1
        b_g1 -= learning_rate * db_g1
        W_g2 -= learning_rate * dW_g2
        b_g2 -= learning_rate * db_g2

    def gen_forward(z_in):
        x_out, h_out = forward_g(z_in)
        return x_out, h_out, z_in

    return gen_forward