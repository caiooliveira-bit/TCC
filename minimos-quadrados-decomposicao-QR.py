import numpy as np
import matplotlib.pyplot as plt

def minimos_quadrados_qr(A, b):
    """
    Resolve o problema de mínimos quadrados min ||Ax - b||_2
    utilizando decomposição QR reduzida.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz de coeficientes (m x n) com m >= n
    b : np.ndarray
        Vetor de observações (m,)
    
    Retorna:
    --------
    x : np.ndarray
        Vetor solução (n,) que minimiza ||Ax - b||_2
    Q1 : np.ndarray
        Matriz Q reduzida (m x n)    R1 : np.ndarray
        Matriz R triangular superior (n x n)
    """
    
    # Validação das dimensões
    m, n = A.shape
    if m < n:
        raise ValueError(f"O sistema é subdeterminado: m={m} < n={n}. "
                         f"É necessário m >= n para mínimos quadrados.")
    
    if len(b) != m:
        raise ValueError(f"Dimensões incompatíveis: A tem {m} linhas, "
                         f"mas b tem {len(b)} elementos.")
    
    # Passo 1: Decomposição QR reduzida
    # Usando numpy.linalg.qr com mode='reduced'
    Q1, R1 = np.linalg.qr(A, mode='reduced')
    
    # Verifica se a decomposição foi bem-sucedida
    if R1.shape[0] != n or R1.shape[1] != n:
        raise ValueError("A decomposição QR reduzida não produziu uma matriz R quadrada.")
    
    # Passo 2: Calcular d = Q1^T * b
    d = Q1.T @ b
    
    # Passo 3: Resolver R1 * x = d por substituição regressiva
    x = resolver_sistema_triangular_superior(R1, d)
    
    return x, Q1, R1

def resolver_sistema_triangular_superior(R, d):
    """
    Resolve um sistema triangular superior Rx = d por substituição regressiva.
    
    Parâmetros:
    -----------
    R : np.ndarray
        Matriz triangular superior (n x n)
    d : np.ndarray
        Vetor independente (n,)
    
    Retorna:
    --------
    x : np.ndarray
        Solução do sistema (n,)
    """
    n = len(d)
    x = np.zeros(n)
    
    # Substituição regressiva: resolve de baixo para cima
    for i in range(n-1, -1, -1):
        if np.abs(R[i, i]) < 1e-12:
            raise ValueError(f"Matriz singular: elemento diagonal R[{i},{i}] ≈ 0")
        
        soma = 0.0
        for j in range(i+1, n):
            soma += R[i, j] * x[j]
        
        x[i] = (d[i] - soma) / R[i, i]
    
    return x

def calcular_residuos(A, b, x):
    """
    Calcula os resíduos e métricas de qualidade do ajuste.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz de coeficientes
    b : np.ndarray
        Vetor de observações
    x : np.ndarray
        Solução encontrada
    
    Retorna:
    --------
    residuos : np.ndarray
        Vetor de resíduos (b - A*x)
    norma_residual : float
        Norma Euclidiana do resíduo
    R2 : float
        Coeficiente de determinação
    """
    b_predito = A @ x
    residuos = b - b_predito
    norma_residual = np.linalg.norm(residuos)
    
    # Cálculo do R² (coeficiente de determinação)
    ss_res = np.sum(residuos**2)
    ss_tot = np.sum((b - np.mean(b))**2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return residuos, norma_residual, R2