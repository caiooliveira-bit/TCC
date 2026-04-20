import numpy as np
import matplotlib.pyplot as plt

def decomposicao_qr_gram_schmidt(A):
    """
    Decomposição QR usando o processo de Gram-Schmidt modificado.
    
    Parâmetros:
    A : matriz m x n (m >= n)
    
    Retorna:
    Q : matriz ortogonal m x n
    R : matriz triangular superior n x n
    """
    m, n = A.shape
    
    if m < n:
        raise ValueError("A matriz deve ter número de linhas >= número de colunas")
    
    # Copia a matriz para não modificar a original
    A = A.astype(float)
    
    # Inicializa Q e R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    # Processo de Gram-Schmidt modificado (mais estável)
    V = A.copy()
    
    for j in range(n):
        # Ortogonalização
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] -= R[i, j] * Q[:, i]
        
        # Normalização
        R[j, j] = np.linalg.norm(V[:, j])
        
        if R[j, j] < 1e-12:
            raise ValueError(f"A matriz é singular ou quase singular (R[{j},{j}] ≈ 0)")
        
        Q[:, j] = V[:, j] / R[j, j]
    
    return Q, R


def decomposicao_qr_householder(A):
    """
    Decomposição QR usando reflexões de Householder.
    Mais estável numericamente que Gram-Schmidt.
    
    Parâmetros:
    A : matriz m x n
    
    Retorna:
    Q : matriz ortogonal m x m
    R : matriz triangular superior m x n
    """
    m, n = A.shape
    A = A.astype(float)
    
    # Copia a matriz
    R = A.copy()
    Q = np.eye(m)
    
    for j in range(min(m-1, n)):
        # Vetor de Householder
        x = R[j:, j]
        
        # Norma do vetor
        norm_x = np.linalg.norm(x)
        
        if norm_x < 1e-12:
            continue
        
        # Sinal para estabilidade numérica
        alpha = -np.sign(x[0]) * norm_x
        
        # Vetor u
        u = x.copy()
        u[0] -= alpha
        
        # Normaliza u
        u = u / np.linalg.norm(u)
        
        # Atualiza R
        R[j:, j:] -= 2 * np.outer(u, np.dot(u.T, R[j:, j:]))
        
        # Atualiza Q
        Q[:, j:] -= 2 * np.outer(np.dot(Q[:, j:], u), u)
    
    return Q, R


def decomposicao_qr_givens(A):
    """
    Decomposição QR usando rotações de Givens.
    Útil para matrizes esparsas.
    
    Parâmetros:
    A : matriz m x n
    
    Retorna:
    Q : matriz ortogonal m x m
    R : matriz triangular superior m x n
    """
    m, n = A.shape
    A = A.astype(float)
    
    # Copia a matriz
    R = A.copy()
    Q = np.eye(m)
    
    for j in range(min(m, n)):
        for i in range(m-1, j, -1):
            if R[i, j] != 0:
                # Calcula rotação de Givens
                a = R[i-1, j]
                b = R[i, j]
                
                r = np.hypot(a, b)
                c = a / r
                s = -b / r
                
                # Matriz de rotação
                G = np.eye(m)
                G[i-1, i-1] = c
                G[i, i] = c
                G[i-1, i] = -s
                G[i, i-1] = s
                
                # Atualiza R
                R = G @ R
                
                # Atualiza Q
                Q = Q @ G.T
    
    return Q, R


def resolver_sistema_qr(A, b, metodo='householder'):
    """
    Resolve o sistema linear Ax = b usando decomposição QR.
    
    Parâmetros:
    A : matriz m x n (m >= n)
    b : vetor de termos independentes
    metodo : 'gram_schmidt', 'householder', ou 'givens'
    
    Retorna:
    x : solução (mínimos quadrados se m > n)
    """
    m, n = A.shape
    
    # Escolhe o método de decomposição
    if metodo == 'gram_schmidt':
        Q, R = decomposicao_qr_gram_schmidt(A)
    elif metodo == 'householder':
        Q, R = decomposicao_qr_householder(A)
    elif metodo == 'givens':
        Q, R = decomposicao_qr_givens(A)
    else:
        raise ValueError("Método deve ser 'gram_schmidt', 'householder' ou 'givens'")
    
    # Se Q é m x m, pegamos apenas as primeiras n colunas
    if Q.shape[1] > n:
        Q = Q[:, :n]
        R = R[:n, :]
    
    # Transforma o vetor b
    b = b.astype(float)
    b_transformed = Q.T @ b
    
    # Resolve sistema triangular superior Rx = b_transformed[:n]
    x = resolver_triangular_superior(R[:n, :n], b_transformed[:n])
    
    return x


def resolver_triangular_superior(R, b):
    """
    Resolve sistema triangular superior Rx = b.
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if abs(R[i, i]) < 1e-12:
            raise ValueError("Matriz singular: elemento diagonal zero")
        
        soma = np.sum(R[i, i+1:] * x[i+1:])
        x[i] = (b[i] - soma) / R[i, i]
    
    return x


def minimos_quadrados_qr(A, b, metodo='householder'):
    """
    Resolve problema de mínimos quadrados usando decomposição QR.
    Minimiza ||Ax - b||₂.
    """
    x = resolver_sistema_qr(A, b, metodo)
    
    # Calcula resíduo
    residuo = np.linalg.norm(A @ x - b)
    
    return x, residuo


def verificar_decomposicao(A, Q, R, tol=1e-10):
    """
    Verifica se a decomposição QR está correta.
    """
    m, n = A.shape
    
    # Verifica Q^T Q = I
    qtq = Q.T @ Q
    erro_ortogonalidade = np.linalg.norm(qtq - np.eye(n))
    
    # Verifica A = QR
    erro_reconstrucao = np.linalg.norm(A - Q @ R)
    
    # Verifica se R é triangular superior
    trilha_superior = np.triu(R)
    erro_triangular = np.linalg.norm(R - trilha_superior)
    
    print(f"Erro de ortogonalidade (Q^T Q - I): {erro_ortogonalidade:.2e}")
    print(f"Erro de reconstrução (A - QR): {erro_reconstrucao:.2e}")
    print(f"Erro de triangularidade (R): {erro_triangular:.2e}")
    
    return (erro_ortogonalidade < tol and 
            erro_reconstrucao < tol and 
            erro_triangular < tol)


def comparar_metodos(A, b):
    """
    Compara os diferentes métodos de decomposição QR.
    """
    print("=" * 70)
    print("COMPARAÇÃO DOS MÉTODOS DE DECOMPOSIÇÃO QR")
    print("=" * 70)
    
    metodos = ['gram_schmidt', 'householder', 'givens']
    
    for metodo in metodos:
        print(f"\nMétodo: {metodo.upper()}")
        print("-" * 40)
        
        try:
            if metodo == 'gram_schmidt':
                Q, R = decomposicao_qr_gram_schmidt(A)
            elif metodo == 'householder':
                Q, R = decomposicao_qr_householder(A)
            else:
                Q, R = decomposicao_qr_givens(A)
            
            print(f"Q shape: {Q.shape}")
            print(f"R shape: {R.shape}")
            
            # Verifica qualidade
            verificar_decomposicao(A, Q, R)
            
            # Resolve sistema
            x = resolver_sistema_qr(A, b, metodo)
            residuo = np.linalg.norm(A @ x - b)
            print(f"Resíduo: {residuo:.2e}")
            
        except Exception as e:
            print(f"Erro: {e}")


def exemplo_visualizacao():
    """
    Exemplo visual da decomposição QR.
    """
    # Cria uma matriz exemplo
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    
    print("Matriz original A:")
    print(A)
    print()
    
    # Decomposição QR
    Q, R = decomposicao_qr_householder(A)
    
    print("Matriz Q (ortogonal):")
    print(Q)
    print()
    
    print("Matriz R (triangular superior):")
    print(R)
    print()
    
    print("Verificação Q^T Q:")
    print(np.round(Q.T @ Q, 10))
    print()
    
    print("Verificação QR:")
    print(np.round(Q @ R, 10))
    print()
    
    # Visualização das matrizes
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Matriz A
    im1 = axes[0].imshow(A, cmap='viridis', aspect='auto')
    axes[0].set_title('Matriz A')
    axes[0].set_xticks(range(A.shape[1]))
    axes[0].set_yticks(range(A.shape[0]))
    plt.colorbar(im1, ax=axes[0])
    
    # Matriz Q
    im2 = axes[1].imshow(Q, cmap='viridis', aspect='auto')
    axes[1].set_title('Matriz Q')
    axes[1].set_xticks(range(Q.shape[1]))
    axes[1].set_yticks(range(Q.shape[0]))
    plt.colorbar(im2, ax=axes[1])
    
    # Matriz R
    im3 = axes[2].imshow(R, cmap='viridis', aspect='auto')
    axes[2].set_title('Matriz R')
    axes[2].set_xticks(range(R.shape[1]))
    axes[2].set_yticks(range(R.shape[0]))
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


# Exemplos de uso
if __name__ == "__main__":
    
    # Exemplo 1: Sistema quadrado
    print("=" * 70)
    print("EXEMPLO 1: Sistema Quadrado")
    print("=" * 70)
    
    A1 = np.array([[4, 1],
                   [1, 3]])
    b1 = np.array([1, 2])
    
    print("Matriz A:")
    print(A1)
    print("Vetor b:", b1)
    
    x1 = resolver_sistema_qr(A1, b1, metodo='householder')
    print(f"Solução: {x1}")
    print(f"Verificação Ax - b: {A1 @ x1 - b1}")
    print()
    
    # Exemplo 2: Sistema retangular (mínimos quadrados)
    print("=" * 70)
    print("EXEMPLO 2: Mínimos Quadrados")
    print("=" * 70)
    
    A2 = np.array([[1, 1],
                   [1, 2],
                   [1, 3]])
    b2 = np.array([2, 3, 4])
    
    print("Matriz A (sobredeterminada):")
    print(A2)
    print("Vetor b:", b2)
    
    x2, residuo = minimos_quadrados_qr(A2, b2)
    print(f"Solução (mínimos quadrados): {x2}")
    print(f"Resíduo: {residuo:.6f}")
    print(f"Verificação: Ax = {A2 @ x2}")
    print()
    
    # Exemplo 3: Matriz mal condicionada
    print("=" * 70)
    print("EXEMPLO 3: Matriz Mal Condicionada")
    print("=" * 70)
    
    # Matriz de Hilbert (mal condicionada)
    n = 5
    A3 = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    b3 = np.ones(n)
    
    print(f"Matriz de Hilbert {n}x{n} (número de condição: {np.linalg.cond(A3):.2e})")
    
    x3_qr = resolver_sistema_qr(A3, b3, metodo='householder')
    x3_direct = np.linalg.solve(A3, b3)
    
    print(f"Solução QR: {x3_qr}")
    print(f"Solução direta: {x3_direct}")
    print(f"Diferença: {np.linalg.norm(x3_qr - x3_direct):.2e}")
    print()
    
    # Exemplo 4: Comparação dos métodos
    print("=" * 70)
    print("EXEMPLO 4: Comparação de Métodos")
    print("=" * 70)
    
    m, n = 10, 5
    A4 = np.random.rand(m, n)
    b4 = np.random.rand(m)
    
    comparar_metodos(A4, b4)
    
    # Visualização
    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÃO...")
    print("=" * 70)
    
    exemplo_visualizacao()
    
    # Exemplo 5: Aproximação de curvas com mínimos quadrados
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Aproximação de Curvas")
    print("=" * 70)
    
    # Dados experimentais
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0.1, 0.9, 2.1, 2.9, 4.2, 4.8])
    
    # Ajuste linear y = a + bx
    A5 = np.column_stack([np.ones(len(x_data)), x_data])
    
    # Resolve por mínimos quadrados usando QR
    params, residuo = minimos_quadrados_qr(A5, y_data)
    
    print(f"Parâmetros ajustados: a = {params[0]:.4f}, b = {params[1]:.4f}")
    print(f"Resíduo: {residuo:.4f}")
    print(f"Reta ajustada: y = {params[0]:.4f} + {params[1]:.4f}x")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, 'bo', label='Dados experimentais')
    
    x_fit = np.linspace(0, 5, 100)
    y_fit = params[0] + params[1] * x_fit
    plt.plot(x_fit, y_fit, 'r-', label=f'Ajuste linear: y = {params[0]:.2f} + {params[1]:.2f}x')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Aproximação por Mínimos Quadrados usando QR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()