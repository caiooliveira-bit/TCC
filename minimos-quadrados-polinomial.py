import numpy as np
import matplotlib.pyplot as plt

def minimos_quadrados_polinomial(x, y, grau):
    """
    Ajusta um polinômio de grau 'grau' aos pontos (x, y) pelo método dos mínimos quadrados.
    
    Parâmetros:
    -----------
    x : array-like
        Coordenadas x dos pontos (m,)
    y : array-like
        Coordenadas y dos pontos (m,)
    grau : int
        Grau do polinômio (n < m)
    
    Retorna:
    --------
    c : np.ndarray
        Coeficientes do polinômio [c0, c1, ..., cn] onde Pn(x) = c0 + c1*x + ... + cn*x^n
    V : np.ndarray
        Matriz de Vandermonde (m x (n+1))
    R2 : float
        Coeficiente de determinação
    """
    
    m = len(x)
    n = grau
    
    # Validações
    if n >= m:
        raise ValueError(f"O grau do polinômio (n={n}) deve ser menor que o número de pontos (m={m})")
    
    # Passo 1: Criar matriz de Vandermonde V (m x (n+1))
    # V[i, j] = x_i^j, onde j = 0, 1, ..., n
    V = np.zeros((m, n + 1))
    for i in range(m):
        for j in range(n + 1):
            V[i, j] = x[i] ** j
    
    # Passo 2: Calcular A = V^T * V
    A = V.T @ V
    
    # Passo 3: Calcular b = V^T * y
    b = V.T @ y
    
    # Passo 4: Resolver o sistema A*c = b (equações normais)
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Se a matriz for singular, usa pseudo-inversa
        print("Matriz singular! Usando pseudo-inversa...")
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Cálculo do R² (coeficiente de determinação)
    y_pred = V @ c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return c, V, R2

def avaliar_polinomio(c, x):
    """
    Avalia o polinômio P(x) = c0 + c1*x + c2*x^2 + ... + cn*x^n nos pontos x.
    
    Parâmetros:
    -----------
    c : np.ndarray
        Coeficientes do polinômio [c0, c1, ..., cn]
    x : array-like
        Pontos onde avaliar o polinômio
    
    Retorna:
    --------
    y : np.ndarray
        Valores do polinômio em x
    """
    n = len(c) - 1
    y = np.zeros_like(x)
    for i in range(n + 1):
        y += c[i] * (x ** i)
    return y