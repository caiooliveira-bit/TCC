import numpy as np
import matplotlib.pyplot as plt

def metodo_bissecao(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    Encontra uma raiz da função f no intervalo [a, b] usando o método da bisseção.
    
    Parâmetros:
    -----------
    f : function
        Função contínua f(x)
    a : float
        Extremo esquerdo do intervalo
    b : float
        Extremo direito do intervalo
    tol : float
        Tolerância (critério de parada)
    max_iter : int
        Número máximo de iterações
    verbose : bool
        Se True, exibe detalhes a cada iteração
    
    Retorna:
    --------
    c : float
        Aproximação da raiz
    historico : list
        Lista com os valores de c em cada iteração
    iteracoes : int
        Número de iterações realizadas
    
    Exceções:
    ---------
    ValueError: Se f(a) e f(b) não tiverem sinais opostos
    """
    
    # Verifica a condição inicial do teorema de Bolzano
    fa = f(a)
    fb = f(b)
    
    if fa * fb >= 0:
        raise ValueError(f"O método da bisseção requer f(a) e f(b) com sinais opostos.\n"
                        f"f({a}) = {fa}, f({b}) = {fb}")
    
    historico = []
    n = 0  # contador de iterações
    
    if verbose:
        print("=" * 70)
        print("MÉTODO DA BISSECÇÃO")
        print("=" * 70)
        print(f"Intervalo inicial: [{a}, {b}]")
        print(f"f({a}) = {fa}")
        print(f"f({b}) = {fb}")
        print(f"Tolerância: {tol}")
        print("-" * 70)
        print(f"{'Iteração':^10} | {'a':^15} | {'b':^15} | {'c':^15} | {'f(c)':^15} | {'Erro':^15}")
        print("-" * 70)
    
    # Critério de parada: largura do intervalo > 2 * tolerância
    while (b - a) > 2 * tol and n < max_iter:
        # Calcula o ponto médio
        c = (a + b) / 2
        fc = f(c)
        
        historico.append(c)
        
        if verbose:
            erro = (b - a) / 2
            print(f"{n+1:^10} | {a:^15.8f} | {b:^15.8f} | {c:^15.8f} | {fc:^15.2e} | {erro:^15.2e}")
        
        # Se encontrou a raiz exata (ou muito próxima)
        if abs(fc) < tol:
            if verbose:
                print("-" * 70)
                print(f"Raiz encontrada em c = {c:.8f} com f(c) = {fc:.2e}")
            return c, historico, n+1
        
        # Decide em qual subintervalo a raiz está
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        
        n += 1
    
    # Após o loop, a raiz é o ponto médio do intervalo final
    c = (a + b) / 2
    historico.append(c)
    
    if verbose:
        print("-" * 70)
        if n >= max_iter:
            print(f"Máximo de iterações ({max_iter}) atingido.")
        print(f"Raiz aproximada: c = {c:.8f}")
        print(f"f(c) = {f(c):.2e}")
        print(f"Erro estimado: {(b - a)/2:.2e}")
        print(f"Iterações realizadas: {n+1}")
    
    return c, historico, n+1

def calcular_erro_teorico(a, b, n):
    """
    Calcula o erro teórico máximo do método da bisseção após n iterações.
    
    Erro <= (b - a) / 2^(n+1)
    """
    return (b - a) / (2 ** (n + 1))