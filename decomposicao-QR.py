import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr as scipy_qr

def householder_reflection(x):
    """
    Calcula o vetor v para a reflexão de Householder que anula os elementos abaixo da diagonal.
    
    Parâmetros:
    -----------
    x : np.ndarray
        Vetor a ser transformado (elementos da coluna a partir da diagonal)
    
    Retorna:
    --------
    v : np.ndarray
        Vetor de Householder (normalizado)
    beta : float
        Fator de escala (2/(v^T v))
    """
    n = len(x)
    
    # Sinal para estabilidade numérica
    sigma = np.sign(x[0]) if x[0] != 0 else 1
    
    # Norma euclidiana do vetor
    norm_x = np.linalg.norm(x)
    
    # Vetor e1 = [1, 0, ..., 0]
    e1 = np.zeros(n)
    e1[0] = 1.0
    
    # v = sign(x1) * ||x|| * e1 + x
    v = x + sigma * norm_x * e1
    
    # Normaliza v
    norm_v = np.linalg.norm(v)
    if norm_v > 1e-12:
        v = v / norm_v
    
    # beta = 2 / (v^T v) = 2 (já que v é normalizado)
    beta = 2.0
    
    return v, beta

def householder_qr(A, verbose=True):
    """
    Decomposição QR via Reflexões de Householder.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz A (m x n) com m >= n
    verbose : bool
        Se True, exibe informações do progresso
    
    Retorna:
    --------
    Q : np.ndarray
        Matriz ortogonal Q (m x m)
    R : np.ndarray
        Matriz triangular superior R (m x n)
    """
    
    m, n = A.shape
    
    if m < n:
        raise ValueError(f"A matriz deve ter m >= n. Shape atual: {m}x{n}")
    
    # Inicialização
    R = A.copy().astype(np.float64)
    Q = np.eye(m, dtype=np.float64)
    
    if verbose:
        print("=" * 80)
        print("DECOMPOSIÇÃO QR VIA REFLEXÕES DE HOUSEHOLDER")
        print("=" * 80)
        print(f"Dimensão da matriz: {m}x{n}")
        print("-" * 80)
        print(f"{'k':^5} | {'Dimensão do bloco':^20} | {'Norma da coluna':^15} | {'Norma residual':^15}")
        print("-" * 80)
    
    for k in range(n):
        # Extrai o vetor x = R[k:m, k] (elementos da diagonal para baixo)
        x = R[k:m, k].copy()
        
        # Calcula o vetor de Householder
        v, beta = householder_reflection(x)
        
        # Aplica a transformação à direita na matriz R
        # R[k:m, k:n] ← R[k:m, k:n] - β v (v^T R[k:m, k:n])
        R_k = R[k:m, k:n]
        v_reshape = v.reshape(-1, 1)
        
        # R = R - β * v * (v^T * R)
        R[k:m, k:n] = R_k - beta * (v_reshape @ (v_reshape.T @ R_k))
        
        # Aplica a transformação à direita na matriz Q
        # Q[1:m, k:m] ← Q[1:m, k:m] - β (Q[1:m, k:m] v) v^T
        Q_k = Q[k:m, :]  # Nota: Q é m x m, aplicamos transformação nas linhas
        Q[k:m, :] = Q_k - beta * ((Q_k @ v_reshape) @ v_reshape.T)
        
        # Garante que os elementos abaixo da diagonal sejam exatamente zero
        if k < m - 1:
            R[k+1:m, k] = 0.0
        
        if verbose and (k % max(1, n//10) == 0 or k < 5):
            norma_coluna = np.linalg.norm(x)
            norma_residual = np.linalg.norm(R[k+1:m, k]) if k < m-1 else 0
            print(f"{k+1:^5} | R[{k+1}:{m}, {k+1}]    | {norma_coluna:^15.6e} | {norma_residual:^15.6e}")
    
    if verbose:
        print("-" * 80)
        print("✅ Decomposição QR concluída!")
        print(f"   Ortogonalidade de Q: {np.linalg.norm(Q.T @ Q - np.eye(m)):.2e}")
        print(f"   Erro de decomposição: {np.linalg.norm(A - Q @ R):.2e}")
    
    return Q, R

def householder_qr_economico(A, verbose=False):
    """
    Versão econômica da decomposição QR via Reflexões de Householder.
    Retorna Q1 (m x n) e R1 (n x n) para o caso m > n.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz A (m x n) com m >= n
    verbose : bool
        Se True, exibe informações
    
    Retorna:
    --------
    Q1 : np.ndarray
        Matriz Q reduzida (m x n) com colunas ortonormais
    R1 : np.ndarray
        Matriz triangular superior R (n x n)
    """
    
    m, n = A.shape
    
    if m < n:
        raise ValueError(f"A matriz deve ter m >= n. Shape atual: {m}x{n}")
    
    # Inicialização
    R = A.copy().astype(np.float64)
    
    # Armazena os vetores de Householder
    householder_vectors = []
    betas = []
    
    for k in range(n):
        x = R[k:m, k].copy()
        v, beta = householder_reflection(x)
        
        householder_vectors.append(v)
        betas.append(beta)
        
        # Aplica transformação em R
        R_k = R[k:m, k:n]
        v_reshape = v.reshape(-1, 1)
        R[k:m, k:n] = R_k - beta * (v_reshape @ (v_reshape.T @ R_k))
        
        if k < m - 1:
            R[k+1:m, k] = 0.0
    
    # Constrói Q1 a partir dos vetores de Householder
    Q1 = np.eye(m, n, dtype=np.float64)
    
    for k in range(n-1, -1, -1):
        v = householder_vectors[k]
        beta = betas[k]
        
        Q_k = Q1[k:m, :]
        v_reshape = v.reshape(-1, 1)
        Q1[k:m, :] = Q_k - beta * ((Q_k @ v_reshape) @ v_reshape.T)
    
    # R1 é a parte triangular superior de R
    R1 = R[:n, :n]
    
    if verbose:
        print(f"QR Econômico: {m}x{n} -> Q1: {Q1.shape}, R1: {R1.shape}")
        print(f"Ortogonalidade: {np.linalg.norm(Q1.T @ Q1 - np.eye(n)):.2e}")
    
    return Q1, R1

def verificar_decomposicao_qr(A, Q, R, tol=1e-10):
    """
    Verifica a qualidade da decomposição QR.
    """
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DA DECOMPOSIÇÃO QR")
    print("=" * 80)
    
    m, n = A.shape
    
    # Verifica se Q é ortogonal (Q^T * Q = I)
    QtQ = Q.T @ Q
    erro_ortogonalidade = np.linalg.norm(QtQ - np.eye(Q.shape[1]))
    print(f"Erro de ortogonalidade (||Q^T Q - I||): {erro_ortogonalidade:.2e}")
    print(f"  Q é ortogonal? {erro_ortogonalidade < tol}")
    
    # Verifica se R é triangular superior
    eh_triangular = True
    for i in range(1, R.shape[0]):
        for j in range(0, min(i, R.shape[1])):
            if abs(R[i, j]) > tol:
                eh_triangular = False
                break
    print(f"R é triangular superior? {eh_triangular}")
    
    # Verifica se A = Q * R
    QR = Q @ R
    erro_decomposicao = np.linalg.norm(A - QR)
    print(f"Erro de decomposição (||A - QR||): {erro_decomposicao:.2e}")
    
    return erro_ortogonalidade, erro_decomposicao

def comparar_com_scipy(A):
    """
    Compara a implementação com a função scipy.linalg.qr.
    """
    print("\n" + "=" * 80)
    print("COMPARAÇÃO COM SCIPY.LINALG.QR")
    print("=" * 80)
    
    # Nossa implementação
    Q_house, R_house = householder_qr(A, verbose=False)
    
    # Implementação do SciPy
    Q_scipy, R_scipy = scipy_qr(A, mode='full')
    
    # Ajusta sinais para comparação (as colunas podem ter sinais opostos)
    # Normaliza sinais das diagonais de R
    for i in range(min(R_house.shape[0], R_scipy.shape[0])):
        if R_house[i, i] * R_scipy[i, i] < 0:
            R_house[i, :] *= -1
            Q_house[:, i] *= -1
    
    erro_Q = np.linalg.norm(np.abs(Q_house) - np.abs(Q_scipy))
    erro_R = np.linalg.norm(R_house - R_scipy)
    
    print(f"Diferença entre Q_householder e Q_scipy: {erro_Q:.2e}")
    print(f"Diferença entre R_householder e R_scipy: {erro_R:.2e}")
    
    return erro_Q, erro_R