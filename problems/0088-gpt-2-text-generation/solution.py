import numpy as np


def load_encoder_hparams_and_params(
    model_size: str = "124M", models_dir: str = "models"
):
    class DummyBPE:

        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [
                self.encoder_dict.get(token, self.encoder_dict["<UNK>"])
                for token in tokens
            ]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join(
                [reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids]
            )

    hparams = {"n_ctx": 1024, "n_head": 2}

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [
            {
                "mlp": {
                    "c_fc": {
                        "w": np.random.rand(10, 20),
                        "b": np.random.rand(20),
                    },
                    "c_proj": {
                        "w": np.random.rand(20, 10),
                        "b": np.random.rand(10),
                    },
                },
                "attn": {
                    "c_attn": {
                        "w": np.random.rand(10, 30),
                        "b": np.random.rand(30),
                    },
                    "c_proj": {
                        "w": np.random.rand(10, 10),
                        "b": np.random.rand(10),
                    },
                },
                "ln_1": {"g": np.ones(10), "b": np.zeros(10)},
                "ln_2": {"g": np.ones(10), "b": np.zeros(10)},
            }
        ],
        "ln_f": {"g": np.ones(10), "b": np.zeros(10)},
    }

    encoder = DummyBPE()
    return encoder, hparams, params


def gelu(x):
    return (
        0.5
        * x
        * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    )


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def mha(x, c_attn, c_proj, n_head):
    x_proj = x @ c_attn["w"] + c_attn["b"]
    q, k, v = np.split(x_proj, 3, axis=-1)

    n_seq, n_embd = x.shape
    d_k = n_embd // n_head

    q = q.reshape(n_seq, n_head, d_k).swapaxes(0, 1)
    k = k.reshape(n_seq, n_head, d_k).swapaxes(0, 1)
    v = v.reshape(n_seq, n_head, d_k).swapaxes(0, 1)

    scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(d_k)

    causal_mask = np.triu(np.ones((n_seq, n_seq)), k=1) * -1e10
    scores = scores + causal_mask

    weights = softmax(scores, axis=-1)
    attn_out = weights @ v

    attn_out = attn_out.swapaxes(0, 1).reshape(n_seq, n_embd)
    return attn_out @ c_proj["w"] + c_proj["b"]


def ffn(x, c_fc, c_proj):
    h = gelu(x @ c_fc["w"] + c_fc["b"])
    return h @ c_proj["w"] + c_proj["b"]


def gpt2(inputs, params, n_head):
    wte = params["wte"][inputs]
    wpe = params["wpe"][: len(inputs)]
    x = wte + wpe

    for block in params["blocks"]:
        x = x + mha(
            layer_norm(x, block["ln_1"]["g"], block["ln_1"]["b"]),
            block["attn"]["c_attn"],
            block["attn"]["c_proj"],
            n_head,
        )
        x = x + ffn(
            layer_norm(x, block["ln_2"]["g"], block["ln_2"]["b"]),
            block["mlp"]["c_fc"],
            block["mlp"]["c_proj"],
        )

    x = layer_norm(x, params["ln_f"]["g"], params["ln_f"]["b"])
    logits = x @ params["wte"].T
    return logits


def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    encoder, hparams, params = load_encoder_hparams_and_params()
    prompt_tokens = encoder.encode(prompt)
    tokens = list(prompt_tokens)

    for _ in range(n_tokens_to_generate):
        logits = gpt2(tokens, params, hparams["n_head"])
        next_token = int(np.argmax(logits[-1]))
        tokens.append(next_token)

    # Decode only the newly generated tokens
    generated_tokens = tokens[len(prompt_tokens) :]
    return encoder.decode(generated_tokens)