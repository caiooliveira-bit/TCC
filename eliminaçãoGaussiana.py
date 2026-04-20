import numpy as np

def eliminacao_gauss(A, b):
    """
    Resolve o sistema linear Ax = b usando o método de eliminação de Gauss.
    
    Parâmetros:
    A : matriz quadrada dos coeficientes (numpy array)
    b : vetor dos termos independentes (numpy array)
    
    Retorna:
    x : vetor solução (numpy array)
    
    Lança:
    ValueError: se a matriz for singular ou o sistema não tiver solução única
    """
    
    # Converte para float para evitar problemas com divisão inteira
    A = A.astype(float)
    b = b.astype(float)
    
    n = len(b)
    
    # Cria uma matriz aumentada [A|b]
    Ab = np.column_stack((A, b))
    
    # Fase de eliminação (triangularização)
    for k in range(n-1):
        # Pivoteamento parcial (opcional, mas recomendado para estabilidade)
        # Encontra o pivô máximo na coluna k
        max_index = np.argmax(np.abs(Ab[k:, k])) + k
        if Ab[max_index, k] == 0:
            raise ValueError(f"Matriz singular: pivô zero na coluna {k}")
        
        # Troca as linhas se necessário
        if max_index != k:
            Ab[[k, max_index]] = Ab[[max_index, k]]
        
        # Eliminação
        for i in range(k+1, n):
            fator = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= fator * Ab[k, k:]
    
    # Verifica se a matriz é singular (último pivô zero)
    if Ab[n-1, n-1] == 0:
        raise ValueError("Matriz singular: sistema sem solução única")
    
    # Fase de substituição regressiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        soma = np.sum(Ab[i, i+1:n] * x[i+1:n])
        x[i] = (Ab[i, n] - soma) / Ab[i, i]
    
    return x


def eliminacao_gauss_sem_pivotamento(A, b):
    """
    Versão sem pivoteamento parcial (menos estável numericamente).
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    Ab = np.column_stack((A, b))
    
    # Eliminação
    for k in range(n-1):
        if Ab[k, k] == 0:
            raise ValueError(f"Pivô zero na posição ({k},{k})")
        
        for i in range(k+1, n):
            fator = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= fator * Ab[k, k:]
    
    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, n] - np.sum(Ab[i, i+1:n] * x[i+1:n])) / Ab[i, i]
    
    return x


def resolver_sistema(A, b, pivotamento=True):
    """
    Função wrapper para resolver sistema linear.
    
    Parâmetros:
    A : matriz dos coeficientes (lista de listas ou numpy array)
    b : vetor dos termos independentes (lista ou numpy array)
    pivotamento : booleano para ativar/desativar pivoteamento parcial
    
    Retorna:
    x : vetor solução
    """
    # Converte para numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Verifica se a matriz é quadrada
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz deve ser quadrada")
    
    # Verifica se as dimensões são compatíveis
    if A.shape[0] != len(b):
        raise ValueError("Dimensões de A e b incompatíveis")
    
    # Escolhe o método
    if pivotamento:
        return eliminacao_gauss(A, b)
    else:
        return eliminacao_gauss_sem_pivotamento(A, b)


# Exemplo de uso e testes
if __name__ == "__main__":
    # Exemplo 1: Sistema simples
    A1 = [[2, 1, -1],
          [-3, -1, 2],
          [-2, 1, 2]]
    b1 = [8, -11, -3]
    
    print("Exemplo 1:")
    print("Matriz A:")
    print(np.array(A1))
    print("Vetor b:", b1)
    
    x1 = resolver_sistema(A1, b1)
    print("Solução x:", x1)
    print("Verificação Ax - b:", np.dot(A1, x1) - b1)
    print()
    
    # Exemplo 2: Sistema 2x2
    A2 = [[3, 2],
          [1, -1]]
    b2 = [7, -1]
    
    print("Exemplo 2:")
    print("Matriz A:")
    print(np.array(A2))
    print("Vetor b:", b2)
    
    x2 = resolver_sistema(A2, b2)
    print("Solução x:", x2)
    print("Verificação Ax - b:", np.dot(A2, x2) - b2)
    print()
    
    # Exemplo 3: Comparação com numpy.linalg.solve
    print("Comparação com numpy.linalg.solve:")
    A3 = np.random.rand(5, 5)
    b3 = np.random.rand(5)
    
    x_gauss = resolver_sistema(A3, b3)
    x_numpy = np.linalg.solve(A3, b3)
    
    print("Solução do Gauss:", x_gauss)
    print("Solução do NumPy:", x_numpy)
    print("Diferença:", np.abs(x_gauss - x_numpy))
    print("Erro máximo:", np.max(np.abs(x_gauss - x_numpy)))