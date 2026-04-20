import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu, solve_triangular

def iteracao_inversa(A, mu, x0=None, tol=1e-10, max_iter=100, verbose=True):
    """
    Iteração Inversa para encontrar o autovalor mais próximo de μ e seu autovetor.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada A (n x n)
    mu : float or complex
        Shift (aproximação do autovalor desejado)
    x0 : np.ndarray, opcional
        Vetor inicial (n,). Se None, usa vetor aleatório
    tol : float
        Tolerância para convergência
    max_iter : int
        Número máximo de iterações
    verbose : bool
        Se True, exibe informações da convergência
    
    Retorna:
    --------
    lambda_aprox : float or complex
        Autovalor mais próximo de μ
    v : np.ndarray
        Autovetor associado (normalizado)
    iteracoes : int
        Número de iterações realizadas
    historico : dict
        Histórico de valores estimados de lambda e erros
    """
    
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz deve ser quadrada")
    
    # Inicialização do vetor x
    if x0 is None:
        x0 = np.random.randn(n)
    else:
        x0 = x0.copy()
    
    # Passo 1: Normalizar x0
    x = x0 / np.linalg.norm(x0, ord=2)
    
    # Passo 2: B = A - μI
    B = A - mu * np.eye(n)
    
    # Passo 3: Decomposição LU de B com pivoteamento
    # Usamos scipy.linalg.lu para maior estabilidade
    P, L, U = lu(B)
    
    # Para resolver B*y = x, resolvemos P*L*U*y = x
    # => L*U*y = P^T * x
    
    historico = {
        'lambda': [],
        'erro': [],
        'norma_residuo': []
    }
    
    if verbose:
        print("=" * 80)
        print("ITERAÇÃO INVERSA (MÉTODO DA POTÊNCIA INVERSA)")
        print("=" * 80)
        print(f"Dimensão da matriz: {n}x{n}")
        print(f"Shift (μ): {mu}")
        print(f"Tolerância: {tol}")
        print(f"Máximo de iterações: {max_iter}")
        print("-" * 80)
        print(f"{'Iteração':^10} | {'λ estimado':^20} | {'|Ay - λx|':^20} | {'Convergência':^15}")
        print("-" * 80)
    
    for k in range(max_iter):
        
        # Resolver B*y = x
        # Primeiro: resolver P*L*z = x  => z = (P*L)^(-1) * x
        # Como P é matriz de permutação, P^T * x permuta as linhas
        Pb = P.T @ x
        
        # Resolver L*z = Pb (forward substitution)
        z = solve_triangular(L, Pb, lower=True)
        
        # Resolver U*y = z (backward substitution)
        y = solve_triangular(U, z, lower=False)
        
        # Passo: Calcular λ = μ + 1/(x^T * y)
        xTy = np.dot(x, y)
        
        if abs(xTy) < 1e-12:
            if verbose:
                print("⚠️ Produto interno x^T*y muito próximo de zero. Parando iteração.")
            break
        
        lambda_aprox = mu + 1.0 / xTy
        
        # Normalizar y para obter novo x
        x_novo = y / np.linalg.norm(y, ord=2)
        
        # Verificar convergência: ||A*y - λ*x|| < ε
        Ay = A @ y
        residuo = Ay - lambda_aprox * x
        norma_residuo = np.linalg.norm(residuo, ord=2)
        
        # Critério de convergência adicional: mudança no autovetor
        mudanca_autovetor = np.linalg.norm(x_novo - x, ord=2)
        
        historico['lambda'].append(lambda_aprox)
        historico['erro'].append(mudanca_autovetor)
        historico['norma_residuo'].append(norma_residuo)
        
        if verbose and (k % 5 == 0 or k < 5 or norma_residuo < 1e-4):
            convergencia = "✓" if norma_residuo < tol else ""
            if isinstance(lambda_aprox, complex):
                lambda_str = f"{lambda_aprox.real:.6f}{lambda_aprox.imag:+.6f}j"
            else:
                lambda_str = f"{lambda_aprox:.10f}"
            print(f"{k+1:^10} | {lambda_str:^20} | {norma_residuo:^20.2e} | {convergencia:^15}")
        
        # Critérios de parada
        if norma_residuo < tol:
            if verbose:
                print("-" * 80)
                print(f"✅ Convergência alcançada após {k+1} iterações")
                print(f"   Autovalor encontrado: {lambda_aprox}")
                print(f"   Norma do resíduo: {norma_residuo:.2e}")
            x = x_novo
            break
        
        # Atualizar x
        x = x_novo
    
    else:
        if verbose:
            print("-" * 80)
            print(f"⚠️ Atingido número máximo de iterações ({max_iter})")
            print(f"   Norma do resíduo final: {norma_residuo:.2e}")
    
    return lambda_aprox, x, k+1, historico

def iteracao_inversa_sem_fatoracao(A, mu, x0=None, tol=1e-10, max_iter=100, verbose=True):
    """
    Versão alternativa da iteração inversa que resolve o sistema linear diretamente.
    Menos eficiente para múltiplas iterações, mas útil para demonstração.
    """
    n = A.shape[0]
    
    if x0 is None:
        x0 = np.random.randn(n)
    
    x = x0 / np.linalg.norm(x0, ord=2)
    B = A - mu * np.eye(n)
    
    historico = {'lambda': [], 'erro': []}
    
    if verbose:
        print("\n--- Versão com resolução direta (sem fatoração prévia) ---")
    
    for k in range(max_iter):
        # Resolve B*y = x diretamente
        y = np.linalg.solve(B, x)
        
        xTy = np.dot(x, y)
        lambda_aprox = mu + 1.0 / xTy
        
        x_novo = y / np.linalg.norm(y, ord=2)
        
        residuo = A @ y - lambda_aprox * x
        norma_residuo = np.linalg.norm(residuo, ord=2)
        
        historico['lambda'].append(lambda_aprox)
        historico['erro'].append(norma_residuo)
        
        if norma_residuo < tol:
            if verbose:
                print(f"Convergência após {k+1} iterações: λ = {lambda_aprox}")
            break
        
        x = x_novo
    
    return lambda_aprox, x, k+1, historico

def encontrar_autovalores_por_shift(A, shifts, tol=1e-10, max_iter=100):
    """
    Encontra múltiplos autovalores usando iteração inversa com diferentes shifts.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada
    shifts : list
        Lista de aproximações iniciais para os autovalores
    tol : float
        Tolerância
    max_iter : int
        Máximo de iterações por shift
    
    Retorna:
    --------
    autovalores : list
        Autovalores encontrados
    autovetores : list
        Autovetores correspondentes
    """
    autovalores = []
    autovetores = []
    
    print("\n" + "=" * 80)
    print("ENCONTRANDO MÚLTIPLOS AUTOVALORES POR ITERAÇÃO INVERSA")
    print("=" * 80)
    
    for i, mu in enumerate(shifts):
        print(f"\n--- Shift {i+1}: μ = {mu} ---")
        try:
            lamb, v, it, _ = iteracao_inversa(A, mu, tol=tol, max_iter=max_iter, verbose=True)
            autovalores.append(lamb)
            autovetores.append(v)
        except Exception as e:
            print(f"  Falha para shift {mu}: {e}")
    
    return autovalores, autovetores