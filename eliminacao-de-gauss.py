import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu

def eliminacao_gauss(A, b, pivoteamento_parcial=True, verbose=True):
    """
    Eliminação de Gauss para resolver sistema linear Ax = b.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz de coeficientes (n x n)
    b : np.ndarray
        Vetor de termos independentes (n,)
    pivoteamento_parcial : bool
        Se True, usa pivoteamento parcial (recomendado)
    verbose : bool
        Se True, exibe os passos da eliminação
    
    Retorna:
    --------
    x : np.ndarray
        Solução do sistema (n,)
    U : np.ndarray
        Matriz triangular superior após eliminação
    historico : dict
        Histórico dos pivôs e multiplicadores
    """
    
    n = len(A)
    
    # Validações
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz A deve ser quadrada")
    if len(b) != n:
        raise ValueError("Os vetores A e b têm dimensões incompatíveis")
    
    # Cria cópias para não modificar as originais
    U = A.copy().astype(np.float64)
    c = b.copy().astype(np.float64)
    
    # Vetor de permutação (registra as trocas de linha)
    permutacao = np.arange(n)
    
    historico = {
        'pivos': [],
        'multiplicadores': [],
        'matrizes': []
    }
    
    if verbose:
        print("=" * 80)
        print("ELIMINAÇÃO DE GAUSS")
        print("=" * 80)
        print(f"Dimensão do sistema: {n}x{n}")
        print(f"Pivoteamento parcial: {'Sim' if pivoteamento_parcial else 'Não'}")
        print("-" * 80)
    
    # Fase 1: Eliminação direta (forward elimination)
    for k in range(n - 1):
        if verbose:
            print(f"\n--- Passo {k+1} (coluna {k+1}) ---")
            print(f"Matriz antes do passo {k+1}:")
            print(np.round(U, 4))
        
        # Pivoteamento parcial
        if pivoteamento_parcial:
            # Encontra o pivô de maior módulo na coluna k a partir da linha k
            pivo_idx = k + np.argmax(np.abs(U[k:, k]))
            
            if pivo_idx != k:
                # Troca as linhas em U e c
                U[[k, pivo_idx]] = U[[pivo_idx, k]]
                c[[k, pivo_idx]] = c[[pivo_idx, k]]
                permutacao[[k, pivo_idx]] = permutacao[[pivo_idx, k]]
                
                if verbose:
                    print(f"  Trocou linha {k+1} com linha {pivo_idx+1}")
        
        # Verifica se a matriz é singular
        if abs(U[k, k]) < 1e-12:
            print(f"⚠️ Aviso: pivô U[{k},{k}] ≈ 0. A matriz pode ser singular.")
            if not pivoteamento_parcial:
                print("   Tente usar pivoteamento parcial.")
        
        historico['pivos'].append(U[k, k])
        
        # Eliminação das linhas abaixo
        multiplicadores = []
        for i in range(k + 1, n):
            # Calcula o multiplicador
            if abs(U[k, k]) > 1e-12:
                fator = U[i, k] / U[k, k]
            else:
                fator = 0
            
            multiplicadores.append(fator)
            
            # Atualiza a linha i
            U[i, k:] -= fator * U[k, k:]
            c[i] -= fator * c[k]
        
        historico['multiplicadores'].append(multiplicadores)
        
        if verbose:
            print(f"  Multiplicadores: {[round(m, 4) for m in multiplicadores]}")
            print(f"Matriz após eliminação:")
            print(np.round(U, 4))
    
    # Fase 2: Substituição regressiva (back substitution)
    x = np.zeros(n)
    
    if verbose:
        print("\n" + "-" * 80)
        print("SUBSTITUIÇÃO REGRESSIVA")
        print("-" * 80)
    
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-12:
            raise ValueError(f"Matriz singular: U[{i},{i}] ≈ 0. Sistema sem solução única.")
        
        soma = 0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        
        x[i] = (c[i] - soma) / U[i, i]
        
        if verbose:
            print(f"  x[{i+1}] = ({c[i]:.4f} - {soma:.4f}) / {U[i, i]:.4f} = {x[i]:.8f}")
    
    if verbose:
        print("\n" + "-" * 80)
        print("RESULTADO FINAL:")
        print(f"Solução x: {x}")
        
        # Verificação
        residuo = b - A @ x
        print(f"Resíduo ||b - Ax||: {np.linalg.norm(residuo):.2e}")
        print("=" * 80)
    
    return x, U, historico

def eliminacao_gauss_aumentada(A_aug, verbose=False):
    """
    Eliminação de Gauss para matriz aumentada.
    
    Parâmetros:
    -----------
    A_aug : np.ndarray
        Matriz aumentada (n x n+1) onde a última coluna é o vetor b
    verbose : bool
        Se True, exibe os passos
    
    Retorna:
    --------
    x : np.ndarray
        Solução do sistema
    """
    n = A_aug.shape[0]
    M = A_aug.copy().astype(np.float64)
    
    for k in range(n - 1):
        # Pivoteamento parcial
        pivo_idx = k + np.argmax(np.abs(M[k:, k]))
        if pivo_idx != k:
            M[[k, pivo_idx]] = M[[pivo_idx, k]]
        
        if abs(M[k, k]) < 1e-12:
            continue
        
        for i in range(k + 1, n):
            fator = M[i, k] / M[k, k]
            M[i, k:] -= fator * M[k, k:]
    
    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(M[i, i]) < 1e-12:
            raise ValueError("Matriz singular")
        x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:n]) / M[i, i]
    
    return x

def eliminacao_gauss_inversa(A, verbose=False):
    """
    Calcula a inversa da matriz A usando eliminação de Gauss.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada A (n x n)
    verbose : bool
        Se True, exibe informações
    
    Retorna:
    --------
    A_inv : np.ndarray
        Matriz inversa de A
    """
    n = A.shape[0]
    
    # Cria a matriz aumentada [A | I]
    A_aug = np.hstack([A.copy().astype(np.float64), np.eye(n)])
    
    if verbose:
        print("\n" + "=" * 80)
        print("CÁLCULO DA MATRIZ INVERSA POR ELIMINAÇÃO DE GAUSS")
        print("=" * 80)
    
    # Fase 1: Triangularização (forward elimination)
    for k in range(n):
        # Pivoteamento parcial
        pivo_idx = k + np.argmax(np.abs(A_aug[k:, k]))
        if pivo_idx != k:
            A_aug[[k, pivo_idx]] = A_aug[[pivo_idx, k]]
            if verbose:
                print(f"  Passo {k+1}: trocou linha {k+1} com linha {pivo_idx+1}")
        
        if abs(A_aug[k, k]) < 1e-12:
            raise ValueError("Matriz singular. Não é possível calcular a inversa.")
        
        # Normaliza a linha do pivô
        pivô = A_aug[k, k]
        A_aug[k, :] /= pivô
        
        # Elimina as outras linhas
        for i in range(n):
            if i != k:
                fator = A_aug[i, k]
                A_aug[i, :] -= fator * A_aug[k, :]
        
        if verbose and (k % max(1, n//5) == 0):
            print(f"  Após passo {k+1}:")
            print(np.round(A_aug, 6))
    
    # A matriz inversa está na parte direita
    A_inv = A_aug[:, n:]
    
    if verbose:
        # Verificação
        erro = np.linalg.norm(A @ A_inv - np.eye(n))
        print(f"\nErro de inversão ||A*A_inv - I||: {erro:.2e}")
    
    return A_inv

def calcular_determinante(A):
    """
    Calcula o determinante de A usando eliminação de Gauss.
    O determinante é o produto dos pivôs (considerando trocas de linha).
    """
    n = A.shape[0]
    U = A.copy().astype(np.float64)
    det = 1.0
    trocas = 0
    
    for k in range(n - 1):
        # Pivoteamento parcial
        p = k + np.argmax(np.abs(U[k:, k]))
        if p != k:
            U[[k, p]] = U[[p, k]]
            trocas += 1
        
        if abs(U[k, k]) < 1e-12:
            return 0.0
        
        det *= U[k, k]
        
        for i in range(k + 1, n):
            fator = U[i, k] / U[k, k]
            U[i, k:] -= fator * U[k, k:]
    
    det *= U[n-1, n-1]
    
    # Cada troca de linha multiplica o determinante por -1
    if trocas % 2 == 1:
        det *= -1
    
    return det

def resolver_sistema_teste():
    """
    Resolve um sistema exemplo para demonstração.
    """
    # Sistema exemplo
    A = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ])
    
    b = np.array([8.0, -11.0, -3.0])
    
    print("\n" + "=" * 80)
    print("SISTEMA EXEMPLO")
    print("=" * 80)
    print("Matriz A:")
    print(A)
    print(f"Vetor b: {b}")
    
    # Solução exata (esperada)
    x_exato = np.linalg.solve(A, b)
    print(f"Solução exata (np.linalg.solve): {x_exato}")
    
    # Nossa implementação
    x, U, _ = eliminacao_gauss(A, b, pivoteamento_parcial=True, verbose=True)
    
    return x