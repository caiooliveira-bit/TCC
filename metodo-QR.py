import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hessenberg

def algoritmo_qr_francis(A, tol=1e-10, max_iter=1000, verbose=True):
    """
    Algoritmo QR com Shifts (Francis) para calcular a forma de Schur de uma matriz.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz quadrada A (n x n)
    tol : float
        Tolerância para convergência
    max_iter : int
        Número máximo de iterações
    verbose : bool
        Se True, exibe informações da convergência
    
    Retorna:
    --------
    T : np.ndarray
        Matriz triangular superior por blocos (forma de Schur)
    Q : np.ndarray
        Matriz ortogonal tal que A = Q * T * Q^T
    iteracoes : int
        Número de iterações realizadas
    historico : list
        Histórico dos valores das subdiagonais
    """
    
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A matriz deve ser quadrada")
    
    # Passo 1: Redução para forma de Hessenberg (pré-processamento)
    # Isso melhora significativamente a eficiência
    if verbose:
        print("Reduzindo matriz para forma de Hessenberg...")
    
    # Usamos scipy para a redução (equivalente ao pré-processamento do algoritmo)
    Q_hess, T_hess = hessenberg(A, calc_q=True)
    
    T = T_hess.copy()
    Q = Q_hess.copy()
    
    historico = []
    iteracao = 0
    
    if verbose:
        print("\n" + "=" * 80)
        print("ALGORITMO QR COM SHIFTS (FRANCIS)")
        print("=" * 80)
        print(f"Dimensão da matriz: {n}x{n}")
        print(f"Tolerância: {tol}")
        print(f"Máximo de iterações: {max_iter}")
        print("-" * 80)
        print(f"{'Iteração':^10} | {'Shift (μ)':^15} | {'Norma subdiag':^15} | {'Status':^20}")
        print("-" * 80)
    
    # Loop principal
    while iteracao < max_iter:
        
        # Verifica convergência: elementos subdiagonais próximos de zero
        norma_subdiag = 0
        for i in range(1, n):
            for j in range(min(i, n-1)):
                if abs(T[i, j]) > tol:
                    norma_subdiag += abs(T[i, j])
        
        historico.append(norma_subdiag)
        
        # Se já convergiu, sai do loop
        if norma_subdiag < tol:
            if verbose:
                print("-" * 80)
                print(f"✅ Convergência alcançada após {iteracao} iterações")
                print(f"   Norma subdiagonal final: {norma_subdiag:.2e}")
            break
        
        # Seleção do shift (estratégia de Francis)
        # Shift de Wilkinson para matrizes reais
        if n >= 2:
            # Extrai o bloco 2x2 inferior direito
            s = T[-2:, -2:]
            
            # Calcula autovalores do bloco 2x2
            # Shift de Wilkinson: escolhe o autovalor mais próximo de T[n-1, n-1]
            a, b, c, d = s[0, 0], s[0, 1], s[1, 0], s[1, 1]
            
            # Discriminante
            disc = (a - d)**2 + 4 * b * c
            if disc >= 0:
                # Autovalores reais
                lambda1 = (a + d + np.sqrt(disc)) / 2
                lambda2 = (a + d - np.sqrt(disc)) / 2
                
                # Escolhe o autovalor mais próximo do elemento (n-1, n-1)
                if abs(lambda1 - T[n-1, n-1]) < abs(lambda2 - T[n-1, n-1]):
                    mu = lambda1
                else:
                    mu = lambda2
            else:
                # Autovalores complexos - usa o elemento (n-1, n-1) como shift
                mu = T[n-1, n-1]
        else:
            # Para matriz 1x1, shift é o próprio elemento
            mu = T[0, 0]
        
        # Passo 2: Calcula decomposição QR de (T - μI)
        T_minus_muI = T - mu * np.eye(n)
        Q_k, R_k = np.linalg.qr(T_minus_muI)
        
        # Passo 3: Atualiza T = R_k * Q_k + μI
        T = R_k @ Q_k + mu * np.eye(n)
        
        # Passo 4: Acumula a matriz ortogonal Q = Q * Q_k
        Q = Q @ Q_k
        
        iteracao += 1
        
        # Exibe progresso a cada 10 iterações
        if verbose and (iteracao % 10 == 0 or iteracao < 5):
            print(f"{iteracao:^10} | {mu:^15.6f} | {norma_subdiag:^15.2e} | {'Processando...':^20}")
    
    if iteracao >= max_iter and norma_subdiag >= tol:
        if verbose:
            print(f"⚠️ Atingido número máximo de iterações ({max_iter})")
            print(f"   Norma subdiagonal atual: {norma_subdiag:.2e}")
    
    # Após convergência, extraímos autovalores da diagonal
    autovalores = np.diag(T).copy()
    
    # Identifica blocos 2x2 para autovalores complexos
    i = 0
    autovalores_complexos = []
    while i < n:
        if i < n-1 and abs(T[i+1, i]) > tol:
            # Bloco 2x2 - autovalores complexos conjugados
            a, b, c, d = T[i, i], T[i, i+1], T[i+1, i], T[i+1, i+1]
            traco = a + d
            det = a*d - b*c
            disc = traco**2 - 4*det
            if disc < 0:  # Par complexo
                lambda1 = complex(traco/2, np.sqrt(-disc)/2)
                lambda2 = complex(traco/2, -np.sqrt(-disc)/2)
                autovalores_complexos.append(lambda1)
                autovalores_complexos.append(lambda2)
                i += 2
                continue
        autovalores_complexos.append(complex(T[i, i], 0))
        i += 1
    
    if verbose:
        print("-" * 80)
        print("\n📊 RESULTADOS FINAIS:")
        print(f"  Forma de Schur T (triangular superior por blocos):")
        print(f"  {T}")
        print(f"\n  Matriz ortogonal Q (Q^T * A * Q = T):")
        print(f"  {Q}")
        print(f"\n  Autovalores encontrados: {autovalores_complexos}")
        print(f"  Número de iterações: {iteracao}")
    
    return T, Q, iteracao, historico

def verificar_forma_schur(A, T, Q, tol=1e-8):
    """
    Verifica se T = Q^T * A * Q é realmente a forma de Schur.
    """
    print("\n" + "=" * 80)
    print("VERIFICAÇÃO DA FORMA DE SCHUR")
    print("=" * 80)
    
    # Verifica se Q é ortogonal
    Q_ortogonal = np.allclose(Q.T @ Q, np.eye(len(Q)), atol=tol)
    print(f"Q é ortogonal? {Q_ortogonal}")
    
    # Verifica se Q^T * A * Q = T
    QtAQ = Q.T @ A @ Q
    forma_correta = np.allclose(QtAQ, T, atol=tol)
    print(f"Q^T * A * Q = T? {forma_correta}")
    
    # Verifica se T é triangular superior por blocos
    n = len(T)
    eh_triangular_superior = True
    for i in range(1, n):
        for j in range(0, i-1):
            if abs(T[i, j]) > tol:
                eh_triangular_superior = False
                break
    
    print(f"T é triangular superior? {eh_triangular_superior}")
    
    # Erro de decomposição
    erro = np.linalg.norm(QtAQ - T)
    print(f"Erro de decomposição: {erro:.2e}")
    
    return Q_ortogonal and forma_correta and eh_triangular_superior

def shift_raizes_quadraticas(T, i, n):
    """
    Implementa shift duplo de Francis para matrizes reais.
    """
    if i == n-2:
        # Último bloco 2x2
        a, b, c, d = T[i, i], T[i, i+1], T[i+1, i], T[i+1, i+1]
        traco = a + d
        det = a*d - b*c
        return traco, det
    else:
        # Usa o bloco 2x2 inferior direito
        a, b, c, d = T[-2, -2], T[-2, -1], T[-1, -2], T[-1, -1]
        traco = a + d
        det = a*d - b*c
        return traco, det