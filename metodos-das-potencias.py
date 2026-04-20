import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs as sparse_eigs

def metodo_potencias(A, x0=None, tol=1e-10, max_iter=100, verbose=True, return_history=True):
    """
    Método das Potências para encontrar o autovalor dominante (maior módulo) e seu autovetor.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada A (n x n)
    x0 : np.ndarray, opcional
        Vetor inicial (n,). Se None, usa vetor aleatório    tol : float
        Tolerância para convergência
    max_iter : int
        Número máximo de iterações
    verbose : bool
        Se True, exibe informações da convergência
    return_history : bool
        Se True, retorna histórico de convergência
    
    Retorna:
    --------
    lambda_dominante : float
        Autovalor dominante (maior em módulo)
    v : np.ndarray
        Autovetor associado (normalizado)
    iteracoes : int
        Número de iterações realizadas
    historico : dict (opcional)
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
    
    historico = {
        'lambda': [],
        'erro_autovalor': [],
        'erro_autovetor': [],
        'norma_y': []
    }
    
    lambda_anterior = 0
    autovalor_exato = None  # Será preenchido se disponível
    
    if verbose:
        print("=" * 80)
        print("MÉTODO DAS POTÊNCIAS")
        print("=" * 80)
        print(f"Dimensão da matriz: {n}x{n}")
        print(f"Tolerância: {tol}")
        print(f"Máximo de iterações: {max_iter}")
        print("-" * 80)
        print(f"{'Iteração':^10} | {'λ (estimado)':^20} | {'|λ_k - λ_{k-1}|':^20} | {'||y - λx||₂':^18}")
        print("-" * 80)
    
    for k in range(1, max_iter + 1):
        # Passo: y = A * x
        y = A @ x
        
        # Passo: λ = x^T * y (Quociente de Rayleigh)
        lambda_atual = np.dot(x, y)
        
        # Passo: x ← y / ||y||₂
        norma_y = np.linalg.norm(y, ord=2)
        
        # Verifica se a norma é zero (matriz nula)
        if norma_y < 1e-12:
            if verbose:
                print("⚠️ Norma de y é zero. Matriz pode ser nula.")
            return 0, x, k, historico
        
        x_novo = y / norma_y
        
        # Critérios de convergência
        erro_autovalor = abs(lambda_atual - lambda_anterior)
        
        # Verificação: ||y - λx||₂
        y_minus_lambda_x = y - lambda_atual * x
        erro_residuo = np.linalg.norm(y_minus_lambda_x, ord=2)
        
        # Mudança no autovetor
        erro_autovetor = np.linalg.norm(x_novo - x, ord=2)
        
        # Armazenar histórico
        historico['lambda'].append(lambda_atual)
        historico['erro_autovalor'].append(erro_autovalor)
        historico['erro_autovetor'].append(erro_autovetor)
        historico['norma_y'].append(norma_y)
        
        if verbose and (k % 5 == 0 or k <= 5 or erro_residuo < 1e-6):
            print(f"{k:^10} | {lambda_atual:^20.10f} | {erro_autovalor:^20.2e} | {erro_residuo:^18.2e}")
        
        # Critério de parada
        if erro_residuo < tol:
            if verbose:
                print("-" * 80)
                print(f"✅ Convergência alcançada após {k} iterações")
                print(f"   Autovalor dominante: {lambda_atual:.12f}")
                print(f"   Norma do resíduo: {erro_residuo:.2e}")
            return lambda_atual, x_novo, k, historico
        
        # Atualizar para próxima iteração
        lambda_anterior = lambda_atual
        x = x_novo
    
    if verbose:
        print("-" * 80)
        print(f"⚠️ Atingido número máximo de iterações ({max_iter})")
        print(f"   Último autovalor estimado: {lambda_atual:.10f}")
        print(f"   Resíduo final: {erro_residuo:.2e}")
    
    return lambda_atual, x, max_iter, historico

def metodo_potencias_com_shift(A, mu, x0=None, tol=1e-10, max_iter=100, verbose=True):
    """
    Método das Potências com Shift (Método da Potência Inversa)
    Encontra o autovalor mais próximo de μ.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada
    mu : float
        Shift
    x0 : np.ndarray, opcional
        Vetor inicial
    tol : float
        Tolerância
    max_iter : int
        Máximo de iterações
    verbose : bool
        Se True, exibe informações
    
    Retorna:
    --------
    lambda_aprox : float
        Autovalor mais próximo de μ
    v : np.ndarray
        Autovetor associado
    iteracoes : int
        Número de iterações
    """
    B = A - mu * np.eye(A.shape[0])
    lambda_B, v, it, _ = metodo_potencias(B, x0=x0, tol=tol, max_iter=max_iter, verbose=verbose)
    lambda_A = mu + 1.0 / lambda_B
    return lambda_A, v, it

def metodo_potencias_para_autovalor_especifico(A, alvo, tipo='mais_proximo', x0=None, tol=1e-10, max_iter=100):
    """
    Encontra autovalor específico usando estratégia de shift.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada
    alvo : float
        Valor alvo para o autovalor
    tipo : str
        'mais_proximo' ou 'menor' (menor em módulo)
    x0 : np.ndarray, opcional
        Vetor inicial
    tol : float
        Tolerância
    max_iter : int
        Máximo de iterações
    
    Retorna:
    --------
    lambda_aprox : float
        Autovalor encontrado
    v : np.ndarray
        Autovetor associado
    """
    if tipo == 'menor':
        # Para encontrar o menor autovalor, aplicamos método das potências à inversa
        return metodo_potencias_com_shift(A, 0, x0=x0, tol=tol, max_iter=max_iter, verbose=False)
    else:
        # Para encontrar o autovalor mais próximo do alvo
        return metodo_potencias_com_shift(A, alvo, x0=x0, tol=tol, max_iter=max_iter, verbose=False)

def comparar_metodos_potencias(A, verbose=True):
    """
    Compara diferentes variações do método das potências.
    """
    resultados = {}
    
    # Autovalores exatos para referência
    autovalores_exatos = np.linalg.eigvals(A)
    lambda_dominante_exato = max(autovalores_exatos, key=abs)
    lambda_menor_exato = min(autovalores_exatos, key=abs)
    
    if verbose:
        print("\n" + "=" * 80)
        print("COMPARAÇÃO DE VARIAÇÕES DO MÉTODO DAS POTÊNCIAS")
        print("=" * 80)
        print(f"Autovalores exatos: {autovalores_exatos}")
        print(f"Autovalor dominante: {lambda_dominante_exato}")
        print(f"Autovalor de menor módulo: {lambda_menor_exato}")
        print("-" * 80)
    
    # Método das potências padrão
    lamb_dom, _, it_dom, _ = metodo_potencias(A, tol=1e-10, max_iter=100, verbose=False)
    resultados['dominante'] = {'valor': lamb_dom, 'iteracoes': it_dom, 'exato': lambda_dominante_exato}
    
    # Método da potência inversa (shift = 0) para menor autovalor
    lamb_menor, _, it_menor = metodo_potencias_com_shift(A, 0, tol=1e-10, max_iter=100, verbose=False)
    resultados['menor'] = {'valor': lamb_menor, 'iteracoes': it_menor, 'exato': lambda_menor_exato}
    
    if verbose:
        print("\n📊 RESULTADOS DA COMPARAÇÃO:")
        print(f"{'Método':^25} | {'Autovalor encontrado':^20} | {'Valor exato':^20} | {'Iterações':^10} | {'Erro':^12}")
        print("-" * 95)
        for key, val in resultados.items():
            erro = abs(val['valor'] - val['exato'])
            print(f"{key:^25} | {val['valor']:^20.10f} | {val['exato']:^20.10f} | {val['iteracoes']:^10} | {erro:^12.2e}")
    
    return resultados
