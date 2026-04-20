import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu as scipy_lu

def decomposicao_lu_pivoteamento(A, verbose=True):
    """
    Decomposição LU com pivoteamento parcial: PA = LU
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada A (n x n)
    verbose : bool
        Se True, exibe informações do progresso
    
    Retorna:
    --------
    L : np.ndarray
        Matriz triangular inferior (com diagonal unitária)
    U : np.ndarray
        Matriz triangular superior
    P : np.ndarray
        Matriz de permutação tal que PA = LU
    """
    
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz deve ser quadrada")
    
    # Inicialização
    U = A.copy().astype(np.float64)
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)
    
    if verbose:
        print("=" * 80)
        print("DECOMPOSIÇÃO LU COM PIVOTEAMENTO PARCIAL")
        print("=" * 80)
        print(f"Dimensão da matriz: {n}x{n}")
        print("-" * 80)
        print(f"{'k':^5} | {'Pivô escolhido':^15} | {'|pivô|':^12} | {'Troca de linhas':^15}")
        print("-" * 80)
    
    for k in range(n - 1):
        # Encontrar o pivô: elemento de maior módulo na coluna k, a partir da linha k
        p = k + np.argmax(np.abs(U[k:, k]))
        
        if verbose:
            print(f"{k+1:^5} | linha {p+1:^8}   | {abs(U[p, k]):^12.6e} | ", end="")
        
        # Trocar linhas k e p em U, L e P
        if p != k:
            # Trocar linhas em U
            U[[k, p], :] = U[[p, k], :]
            # Trocar linhas em L (apenas a partir da coluna 1 até k-1)
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
            # Trocar linhas em P
            P[[k, p], :] = P[[p, k], :]
            
            if verbose:
                print(f"trocou linha {k+1} com linha {p+1}")
        else:
            if verbose:
                print(f"nenhuma troca")
        
        # Verificar se a matriz é singular
        if abs(U[k, k]) < 1e-12:
            if verbose:
                print(f"\n⚠️ Aviso: U[{k+1},{k+1}] ≈ 0. Matriz pode ser singular.")
            # Continua, mas a decomposição pode ser instável
        
        # Calcular os multiplicadores e atualizar U
        for i in range(k + 1, n):
            # Multiplicador
            L[i, k] = U[i, k] / U[k, k]
            
            # Atualizar a linha i de U
            for j in range(k, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j]
    
    if verbose:
        print("-" * 80)
        print("✅ Decomposição LU concluída!")
        
        # Verificar se PA = LU
        PA = P @ A
        LU = L @ U
        erro = np.linalg.norm(PA - LU)
        print(f"   Erro da decomposição (||PA - LU||): {erro:.2e}")
    
    return L, U, P

def decomposicao_lu_sem_pivoteamento(A, verbose=False):
    """
    Decomposição LU sem pivoteamento (instável para certas matrizes).
    Apenas para comparação.
    """
    n = A.shape[0]
    U = A.copy().astype(np.float64)
    L = np.eye(n, dtype=np.float64)
    
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-12:
            if verbose:
                print(f"⚠️ Pivô zero em U[{k},{k}]. A decomposição falhou.")
            return None, None, None
        
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j]
    
    return L, U, np.eye(n)

def resolver_sistema_lu(L, U, P, b):
    """
    Resolve o sistema linear Ax = b usando a decomposição LU: PA = LU.
    
    Passos:
    1. Permutar b: Pb = P * b
    2. Resolver L*y = Pb (forward substitution)
    3. Resolver U*x = y (backward substitution)
    """
    n = len(b)
    
    # Passo 1: Permutar o vetor b
    Pb = P @ b
    
    # Passo 2: Forward substitution - resolver L*y = Pb
    y = np.zeros(n)
    for i in range(n):
        soma = 0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = Pb[i] - soma
    
    # Passo 3: Backward substitution - resolver U*x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError(f"Matriz singular: U[{i},{i}] ≈ 0")
        soma = 0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    
    return x

def fatoracao_lu_otimizada(A):
    """
    Versão otimizada da decomposição LU usando operações vetorizadas do NumPy.
    """
    n = A.shape[0]
    U = A.copy().astype(np.float64)
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)
    
    for k in range(n - 1):
        # Pivoteamento parcial
        p = k + np.argmax(np.abs(U[k:, k]))
        
        if p != k:
            U[[k, p], :] = U[[p, k], :]
            L[[k, p], :k] = L[[p, k], :k]
            P[[k, p], :] = P[[p, k], :]
        
        if abs(U[k, k]) < 1e-14:
            continue
        
        # Multiplicadores (vetorizado)
        L[k+1:n, k] = U[k+1:n, k] / U[k, k]
        
        # Atualização de U (vetorizada)
        for i in range(k+1, n):
            U[i, k:n] -= L[i, k] * U[k, k:n]
    
    return L, U, P

def comparar_com_scipy(A):
    """
    Compara a implementação com a função scipy.linalg.lu.
    """
    print("\n" + "=" * 80)
    print("COMPARAÇÃO COM SCIPY.LINALG.LU")
    print("=" * 80)
    
    # Nossa implementação
    L_ours, U_ours, P_ours = decomposicao_lu_pivoteamento(A, verbose=False)
    
    # Implementação do SciPy
    P_scipy, L_scipy, U_scipy = scipy_lu(A)
    
    # Ajusta sinais para comparação (Scipy retorna L com diagonal não necessariamente unitária)
    # Normaliza para comparação justa
    for i in range(len(A)):
        if L_ours[i, i] != 1:
            # Ajusta L_ours para ter diagonal unitária
            L_ours[i, :] /= L_ours[i, i]
    
    erro_L = np.linalg.norm(np.abs(L_ours) - np.abs(L_scipy))
    erro_U = np.linalg.norm(np.abs(U_ours) - np.abs(U_scipy))
    
    print(f"Diferença entre L_ours e L_scipy: {erro_L:.2e}")
    print(f"Diferença entre U_ours e U_scipy: {erro_U:.2e}")
    
    return erro_L, erro_U