import numpy as np

def decomposicao_lu_pivotamento(A):
    """
    Realiza a decomposição LU com pivotamento parcial.
    
    Parâmetros:
    A: matriz quadrada (numpy array)
    
    Retorna:
    L: matriz triangular inferior
    U: matriz triangular superior
    P: matriz de permutação
    """
    n = len(A)
    U = A.copy().astype(float)
    L = np.eye(n)
    P = np.eye(n)
    
    for i in range(n):
        # Pivotamento: encontrar a maior linha abaixo de i
        max_index = i + np.argmax(np.abs(U[i:, i]))
        
        if max_index != i:
            # Trocar linhas em U
            U[[i, max_index]] = U[[max_index, i]]
            # Trocar linhas em P
            P[[i, max_index]] = P[[max_index, i]]
            # Trocar linhas em L (apenas para i > 0)
            if i > 0:
                L[[i, max_index], :i] = L[[max_index, i], :i]
        
        # Eliminação
        for j in range(i + 1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] -= L[j, i] * U[i, i:]
    
    return L, U, P

def resolver_lu_pivotamento(A, b):
    """
    Resolve o sistema linear Ax = b usando LU com pivotamento.
    """
    L, U, P = decomposicao_lu_pivotamento(A)
    
    # Aplicar permutação em b
    b_permutado = np.dot(P, b)
    n = len(A)
    
    # Resolver Ly = Pb (substituição progressiva)
    y = np.zeros(n)
    for i in range(n):
        soma = 0
        for j in range(i):
            soma += L[i][j] * y[j]
        y[i] = b_permutado[i] - soma
    
    # Resolver Ux = y (substituição regressiva)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += U[i][j] * x[j]
        x[i] = (y[i] - soma) / U[i][i]
    
    return x
