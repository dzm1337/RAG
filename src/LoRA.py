import numpy as np

alpha = 2.0


def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B) * alpha  # use scaled_lora weights
    return h


np.random.seed(42)

x = np.array([[1.0, 2.0, 3.0]])

W = np.random.randn(3, 4)

W_A = np.random.randn(3, 2)
W_B = np.random.randn(2, 4)

output = lora_forward_matmul(x, W, W_A, W_B)

print("--- Resultados ---")
print("Entrada (x):", x.shape)
print("Saída da camada com LoRA:", output)
print("Formato da saída:", output.shape)
