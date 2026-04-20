import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Union, List, Optional
from dataclasses import dataclass
import warnings

# ============================================================================
# IMPLEMENTAÇÃO BÁSICA DO MÉTODO DE NEWTON-RAPHSON
# ============================================================================

def newton_raphson(f: Callable, 
                   df: Callable, 
                   x0: float, 
                   tol: float = 1e-8, 
                   max_iter: int = 100,
                   verbose: bool = True) -> Tuple[float, int, List[dict]]:
    """
    Método de Newton-Raphson para encontrar raízes de funções.
    
    Parâmetros:
    f : função objetivo f(x)
    df : derivada da função f'(x)
    x0 : chute inicial
    tol : tolerância para o critério de parada
    max_iter : número máximo de iterações
    verbose : se True, mostra informações detalhadas
    
    Retorna:
    x_root : aproximação da raiz
    iterations : número de iterações realizadas
    history : histórico das iterações
    """
    x = float(x0)
    history = []
    
    if verbose:
        print(f"{'Iter':^6} {'x':^15} {'f(x)':^15} {'f\'(x)':^15} {'Erro':^15}")
        print("-" * 75)
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        # Verifica derivada zero
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivada zero em x = {x}. Método não pode continuar.")
        
        # Calcula próximo x
        x_new = x - fx / dfx
        erro = abs(x_new - x)
        
        # Armazena histórico
        history.append({
            'iter': i + 1,
            'x': x,
            'fx': fx,
            'dfx': dfx,
            'x_new': x_new,
            'erro': erro
        })
        
        if verbose:
            print(f"{i+1:^6} {x:^15.8f} {fx:^15.2e} {dfx:^15.2e} {erro:^15.2e}")
        
        # Critério de parada
        if erro < tol or abs(fx) < tol:
            if verbose:
                print(f"\nConvergência alcançada após {i+1} iterações")
                print(f"Raiz aproximada: {x_new:.10f}")
                print(f"f(x) = {f(x_new):.2e}")
            return x_new, i + 1, history
        
        x = x_new
    
    warnings.warn(f"Número máximo de iterações ({max_iter}) atingido sem convergência")
    return x, max_iter, history


def newton_raphson_sem_derivada(f: Callable, 
                                x0: float, 
                                tol: float = 1e-8, 
                                max_iter: int = 100,
                                h: float = 1e-7) -> Tuple[float, int]:
    """
    Método de Newton-Raphson que aproxima a derivada numericamente.
    
    Parâmetros:
    f : função objetivo
    x0 : chute inicial
    tol : tolerância
    max_iter : máximo de iterações
    h : passo para diferença finita
    
    Retorna:
    x_root : aproximação da raiz
    iterations : número de iterações
    """
    x = float(x0)
    
    for i in range(max_iter):
        fx = f(x)
        
        # Aproxima derivada por diferença central
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivada aproximada zero em x = {x}")
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol or abs(fx) < tol:
            return x_new, i + 1
        
        x = x_new
    
    return x, max_iter


# ============================================================================
# MÉTODO DE NEWTON-RAPHSON MODIFICADO
# ============================================================================

def newton_raphson_modificado(f: Callable, 
                              df: Callable, 
                              x0: float, 
                              tol: float = 1e-8, 
                              max_iter: int = 100,
                              fator_relaxacao: float = 1.0) -> Tuple[float, int]:
    """
    Método de Newton-Raphson com fator de relaxação.
    
    Parâmetros:
    fator_relaxacao : fator de relaxação (0 < ω ≤ 1)
                      ω=1: método padrão
                      ω<1: mais estável para funções problemáticas
    """
    x = float(x0)
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivada zero em x = {x}")
        
        # Aplica fator de relaxação
        x_new = x - fator_relaxacao * fx / dfx
        
        if abs(x_new - x) < tol or abs(fx) < tol:
            return x_new, i + 1
        
        x = x_new
    
    return x, max_iter


def newton_raphson_multiplas_raizes(f: Callable, 
                                    df: Callable, 
                                    d2f: Callable,
                                    x0: float, 
                                    tol: float = 1e-8, 
                                    max_iter: int = 100) -> Tuple[float, int]:
    """
    Método de Newton-Raphson modificado para raízes múltiplas.
    
    Usa: x_{n+1} = x_n - (f * f') / (f'^2 - f * f'')
    """
    x = float(x0)
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        
        denominador = dfx**2 - fx * d2fx
        
        if abs(denominador) < 1e-15:
            raise ValueError(f"Denominador zero em x = {x}")
        
        x_new = x - (fx * dfx) / denominador
        
        if abs(x_new - x) < tol or abs(fx) < tol:
            return x_new, i + 1
        
        x = x_new
    
    return x, max_iter


# ============================================================================
# MÉTODO DE NEWTON-RAPHSON PARA SISTEMAS NÃO LINEARES
# ============================================================================

def newton_raphson_sistema(F: List[Callable], 
                           J: Callable, 
                           x0: np.ndarray, 
                           tol: float = 1e-8, 
                           max_iter: int = 100) -> Tuple[np.ndarray, int]:
    """
    Método de Newton-Raphson para sistemas não lineares.
    
    Parâmetros:
    F : lista de funções [f1(x), f2(x), ..., fn(x)]
    J : matriz Jacobiana (função que retorna matriz n x n)
    x0 : vetor inicial
    tol : tolerância
    max_iter : máximo de iterações
    
    Retorna:
    x : solução do sistema
    iterations : número de iterações
    """
    x = np.array(x0, dtype=float)
    n = len(F)
    
    for i in range(max_iter):
        # Avalia funções
        Fx = np.array([f(x) for f in F])
        
        # Calcula Jacobiana
        Jx = J(x)
        
        # Resolve sistema linear Jx * Δx = -Fx
        try:
            delta = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            # Se matriz singular, usa pseudo-inversa
            delta = np.linalg.lstsq(Jx, -Fx, rcond=None)[0]
        
        x_new = x + delta
        
        # Critério de parada
        if np.linalg.norm(delta) < tol or np.linalg.norm(Fx) < tol:
            return x_new, i + 1
        
        x = x_new
    
    return x, max_iter


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE CONVERGÊNCIA
# ============================================================================

def analise_convergencia(history: List[dict]) -> dict:
    """
    Analisa a convergência do método de Newton-Raphson.
    
    Parâmetros:
    history : histórico retornado por newton_raphson
    
    Retorna:
    dict : métricas de convergência
    """
    erros = [h['erro'] for h in history]
    
    # Taxa de convergência
    taxas = []
    for i in range(2, len(erros)):
        if erros[i-1] > 0:
            taxa = np.log(erros[i]) / np.log(erros[i-1])
            taxas.append(taxa)
    
    # Ordem de convergência (estimada)
    if len(taxas) > 0:
        ordem_media = np.mean(taxas)
    else:
        ordem_media = np.nan
    
    # Fator de convergência assintótica
    fatores = []
    for i in range(1, len(erros)):
        if erros[i-1] > 0:
            fatores.append(erros[i] / erros[i-1])
    
    resultados = {
        'erros': erros,
        'taxas_convergencia': taxas,
        'ordem_media': ordem_media,
        'fator_medio': np.mean(fatores) if fatores else np.nan,
        'convergencia_quadratica': ordem_media > 1.8 if not np.isnan(ordem_media) else False,
        'iteracoes': len(history),
        'erro_final': erros[-1] if erros else np.nan
    }
    
    return resultados


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def plot_convergencia(history: List[dict], f: Callable, x_range: Tuple[float, float] = None):
    """
    Plota a convergência do método de Newton-Raphson.
    """
    if not history:
        print("Histórico vazio")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extrai dados
    iteracoes = [h['iter'] for h in history]
    x_vals = [h['x'] for h in history]
    fx_vals = [h['fx'] for h in history]
    erros = [h['erro'] for h in history]
    
    # Gráfico 1: Evolução de x
    ax1 = axes[0, 0]
    ax1.plot(iteracoes, x_vals, 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=x_vals[-1], color='r', linestyle='--', label=f'Raiz ≈ {x_vals[-1]:.6f}')
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('x')
    ax1.set_title('Evolução da Aproximação')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Evolução de f(x)
    ax2 = axes[0, 1]
    ax2.semilogy(iteracoes, np.abs(fx_vals), 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('|f(x)| (log)')
    ax2.set_title('Evolução do Resíduo')
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Erro
    ax3 = axes[1, 0]
    ax3.semilogy(iteracoes, erros, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteração')
    ax3.set_ylabel('Erro (log)')
    ax3.set_title('Convergência do Erro')
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Função e iterações
    ax4 = axes[1, 1]
    if x_range is None:
        x_min = min(x_vals) - 0.5 * (max(x_vals) - min(x_vals))
        x_max = max(x_vals) + 0.5 * (max(x_vals) - min(x_vals))
        x_range = (x_min, x_max)
    
    x_plot = np.linspace(x_range[0], x_range[1], 500)
    y_plot = f(x_plot)
    
    ax4.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plota iterações
    for i, x in enumerate(x_vals):
        ax4.plot(x, f(x), 'ro', markersize=8)
        if i < len(x_vals) - 1:
            # Linha vertical
            ax4.plot([x, x], [f(x), 0], 'r--', alpha=0.5)
            # Linha tangente (opcional)
            if 'dfx' in history[i]:
                df = history[i]['dfx']
                x_tang = np.array([x - 0.5, x + 0.5])
                y_tang = f(x) + df * (x_tang - x)
                ax4.plot(x_tang, y_tang, 'g--', alpha=0.3)
    
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Método de Newton-Raphson - Visualização')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_ordem_convergencia(history: List[dict]):
    """
    Plota a ordem de convergência do método.
    """
    if len(history) < 3:
        print("Precisa de pelo menos 3 iterações para análise de ordem")
        return
    
    erros = np.array([h['erro'] for h in history])
    erros = erros[erros > 0]  # Remove zeros
    
    if len(erros) < 3:
        print("Erros muito pequenos para análise")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico 1: log(erro_n+1) vs log(erro_n)
    ax1 = axes[0]
    log_e_n = np.log(erros[:-1])
    log_e_n1 = np.log(erros[1:])
    
    ax1.plot(log_e_n, log_e_n1, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('log(erro_n)')
    ax1.set_ylabel('log(erro_{n+1})')
    ax1.set_title('Ordem de Convergência')
    
    # Regressão linear para estimar ordem
    if len(log_e_n) > 1:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_e_n, log_e_n1)
        ax1.plot(log_e_n, slope * log_e_n + intercept, 'r--', 
                label=f'Ordem estimada = {slope:.3f}')
        ax1.legend()
    
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Erro vs Iteração (escala log-log)
    ax2 = axes[1]
    iteracoes = np.arange(1, len(erros) + 1)
    ax2.loglog(iteracoes, erros, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Erro')
    ax2.set_title('Convergência (escala log-log)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXEMPLOS E APLICAÇÕES
# ============================================================================

def exemplo_raiz_quadrada():
    """Exemplo: Encontrar √2 usando Newton-Raphson."""
    print("=" * 70)
    print("EXEMPLO 1: Encontrando √2 (raiz de f(x) = x² - 2)")
    print("=" * 70)
    
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    
    # Testa diferentes chutes iniciais
    chutes = [1.0, 1.5, 10.0]
    
    for x0 in chutes:
        print(f"\nChute inicial: x0 = {x0}")
        raiz, iteracoes, history = newton_raphson(f, df, x0, tol=1e-12, verbose=False)
        print(f"Raiz encontrada: {raiz:.12f}")
        print(f"Valor exato: {np.sqrt(2):.12f}")
        print(f"Erro: {abs(raiz - np.sqrt(2)):.2e}")
        print(f"Iterações: {iteracoes}")
        
        # Análise de convergência
        analise = analise_convergencia(history)
        print(f"Convergência quadrática: {analise['convergencia_quadratica']}")
        print(f"Ordem média: {analise['ordem_media']:.3f}")
    
    # Visualização
    raiz, iteracoes, history = newton_raphson(f, df, 1.5, tol=1e-10, verbose=False)
    plot_convergencia(history, f, x_range=(0, 3))
    plot_ordem_convergencia(history)
    
    return raiz


def exemplo_funcao_transcendental():
    """Exemplo: Encontrar raiz de função transcendental."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: f(x) = cos(x) - x")
    print("=" * 70)
    
    f = lambda x: np.cos(x) - x
    df = lambda x: -np.sin(x) - 1
    
    # Diferentes chutes iniciais
    chutes = [0.5, 1.0, 2.0]
    
    for x0 in chutes:
        print(f"\nChute inicial: x0 = {x0}")
        try:
            raiz, iteracoes, history = newton_raphson(f, df, x0, tol=1e-12, verbose=False)
            print(f"Raiz encontrada: {raiz:.12f}")
            print(f"Valor exato: 0.7390851332151606")
            print(f"f(raiz) = {f(raiz):.2e}")
            print(f"Iterações: {iteracoes}")
        except ValueError as e:
            print(f"Erro: {e}")
    
    # Com aproximação numérica da derivada
    print("\nCom aproximação numérica da derivada:")
    raiz, iteracoes = newton_raphson_sem_derivada(f, 0.5, tol=1e-12)
    print(f"Raiz: {raiz:.12f}, Iterações: {iteracoes}")
    
    # Visualização
    raiz, iteracoes, history = newton_raphson(f, df, 0.5, tol=1e-10, verbose=False)
    plot_convergencia(history, f, x_range=(-1, 2))
    
    return raiz


def exemplo_raiz_multipla():
    """Exemplo: Raiz múltipla (convergência lenta)."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Raiz Múltipla - f(x) = (x-1)³")
    print("=" * 70)
    
    f = lambda x: (x-1)**3
    df = lambda x: 3*(x-1)**2
    d2f = lambda x: 6*(x-1)
    
    print("Método padrão de Newton-Raphson (convergência lenta):")
    raiz, iteracoes, history = newton_raphson(f, df, 2.0, tol=1e-12, verbose=False)
    print(f"Raiz: {raiz:.12f}, Iterações: {iteracoes}")
    
    print("\nMétodo modificado para raízes múltiplas (convergência mais rápida):")
    raiz, iteracoes = newton_raphson_multiplas_raizes(f, df, d2f, 2.0, tol=1e-12)
    print(f"Raiz: {raiz:.12f}, Iterações: {iteracoes}")
    
    # Comparação visual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, metodo, titulo in zip(axes, 
                                   ['padrao', 'modificado'],
                                   ['Newton-Raphson Padrão', 'Newton-Raphson Modificado']):
        if metodo == 'padrao':
            _, _, hist = newton_raphson(f, df, 2.0, tol=1e-10, verbose=False)
        else:
            # Para o modificado, precisamos adaptar o histórico
            hist = []
            x = 2.0
            for i in range(20):
                fx = f(x)
                dfx = df(x)
                d2fx = d2f(x)
                denominador = dfx**2 - fx * d2fx
                x_new = x - (fx * dfx) / denominador
                hist.append({'iter': i+1, 'x': x, 'erro': abs(x_new - x)})
                x = x_new
        
        erros = [h['erro'] for h in hist]
        ax.semilogy(range(1, len(erros)+1), erros, 'bo-', linewidth=2)
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Erro (log)')
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return raiz


def exemplo_sistema_nao_linear():
    """Exemplo: Sistema de equações não lineares."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Sistema Não Linear")
    print("=" * 70)
    
    # Sistema:
    # f1(x,y) = x² + y² - 4 = 0  (círculo raio 2)
    # f2(x,y) = x - y² = 0       (parábola)
    
    def F1(x):
        return x[0]**2 + x[1]**2 - 4
    
    def F2(x):
        return x[0] - x[1]**2
    
    F = [F1, F2]
    
    def Jacobiana(x):
        return np.array([
            [2*x[0], 2*x[1]],
            [1, -2*x[1]]
        ])
    
    # Diferentes chutes iniciais
    chutes = [[1, 1], [2, 0], [-1, 1]]
    
    for x0 in chutes:
        print(f"\nChute inicial: {x0}")
        solucao, iteracoes = newton_raphson_sistema(F, Jacobiana, x0, tol=1e-10)
        print(f"Solução: x = {solucao[0]:.8f}, y = {solucao[1]:.8f}")
        print(f"Resíduos: f1 = {F1(solucao):.2e}, f2 = {F2(solucao):.2e}")
        print(f"Iterações: {iteracoes}")
    
    # Visualização
    x_plot = np.linspace(-2.5, 2.5, 100)
    y_plot = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    
    Z1 = X**2 + Y**2 - 4
    Z2 = X - Y**2
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='Círculo')
    plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='Parábola')
    
    # Soluções encontradas
    solucoes = []
    for x0 in chutes:
        sol, _ = newton_raphson_sistema(F, Jacobiana, x0, tol=1e-10)
        solucoes.append(sol)
        plt.plot(sol[0], sol[1], 'go', markersize=10, label=f'Solução {len(solucoes)}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sistema Não Linear - Pontos de Interseção')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    
    return solucoes


def exemplo_otimizacao():
    """Exemplo: Encontrando máximo/mínimo usando Newton-Raphson."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Otimização - Encontrando Máximo de f(x)")
    print("=" * 70)
    
    # Função: f(x) = -x² + 4x + 1 (máximo em x = 2)
    f = lambda x: -x**2 + 4*x + 1
    df = lambda x: -2*x + 4  # Primeira derivada
    d2f = lambda x: -2       # Segunda derivada
    
    # Para encontrar máximo, resolvemos f'(x) = 0
    raiz, iteracoes, history = newton_raphson(df, d2f, 0.0, tol=1e-12, verbose=False)
    
    print(f"Máximo encontrado em x = {raiz:.10f}")
    print(f"Valor máximo: f(x) = {f(raiz):.10f}")
    print(f"Valor exato: x = 2, f(x) = 5")
    
    # Visualização
    x_plot = np.linspace(-1, 5, 200)
    y_plot = f(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
    plt.plot(raiz, f(raiz), 'ro', markersize=10, label=f'Máximo em x = {raiz:.3f}')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=raiz, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Otimização usando Newton-Raphson')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return raiz


def exemplo_comparacao_metodos():
    """Compara Newton-Raphson com outros métodos."""
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Comparação com outros métodos numéricos")
    print("=" * 70)
    
    f = lambda x: x**3 - x - 2
    df = lambda x: 3*x**2 - 1
    
    # Método da bisseção
    def bissecao(f, a, b, tol):
        for i in range(100):
            c = (a + b) / 2
            if abs(f(c)) < tol or (b - a)/2 < tol:
                return c, i + 1
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c, 100
    
    # Método da secante
    def secante(f, x0, x1, tol, max_iter=100):
        for i in range(max_iter):
            fx0, fx1 = f(x0), f(x1)
            if abs(fx1) < tol:
                return x1, i + 1
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            x0, x1 = x1, x_new
        return x1, max_iter
    
    # Comparação
    raiz_exata = 1.5213797068045676
    
    # Newton-Raphson
    raiz_nr, iter_nr, _ = newton_raphson(f, df, 1.5, tol=1e-12, verbose=False)
    
    # Bisseção
    raiz_bis, iter_bis = bissecao(f, 1, 2, 1e-12)
    
    # Secante
    raiz_sec, iter_sec = secante(f, 1, 2, 1e-12)
    
    print(f"{'Método':<20} {'Raiz':<20} {'Erro':<15} {'Iterações':<10}")
    print("-" * 65)
    print(f"{'Newton-Raphson':<20} {raiz_nr:<20.12f} {abs(raiz_nr - raiz_exata):<15.2e} {iter_nr:<10}")
    print(f"{'Bisseção':<20} {raiz_bis:<20.12f} {abs(raiz_bis - raiz_exata):<15.2e} {iter_bis:<10}")
    print(f"{'Secante':<20} {raiz_sec:<20.12f} {abs(raiz_sec - raiz_exata):<15.2e} {iter_sec:<10}")
    
    # Gráfico comparativo
    plt.figure(figsize=(10, 6))
    
    # Função
    x_plot = np.linspace(1, 2, 200)
    y_plot = f(x_plot)
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x³ - x - 2')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Raízes encontradas
    plt.plot(raiz_nr, 0, 'ro', markersize=10, label=f'Newton-Raphson (iter={iter_nr})')
    plt.plot(raiz_bis, 0, 'gs', markersize=10, label=f'Bisseção (iter={iter_bis})')
    plt.plot(raiz_sec, 0, 'md', markersize=10, label=f'Secante (iter={iter_sec})')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Comparação de Métodos Numéricos para Encontrar Raízes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# CLASSE PARA NEWTON-RAPHSON
# ============================================================================

class NewtonRaphsonSolver:
    """
    Classe para resolver equações usando o método de Newton-Raphson.
    """
    
    def __init__(self, f: Callable, df: Optional[Callable] = None, 
                 tol: float = 1e-8, max_iter: int = 100):
        """
        Parâmetros:
        f : função objetivo
        df : derivada (se None, usa aproximação numérica)
        tol : tolerância
        max_iter : máximo de iterações
        """
        self.f = f
        self.df = df if df is not None else self._derivada_numerica
        self.tol = tol
        self.max_iter = max_iter
        self.history = []
        self.root_ = None
        self.iterations_ = None
        
    def _derivada_numerica(self, x: float, h: float = 1e-7) -> float:
        """Aproxima derivada por diferença central."""
        return (self.f(x + h) - self.f(x - h)) / (2 * h)
    
    def solve(self, x0: float, verbose: bool = False) -> float:
        """
        Resolve a equação f(x) = 0.
        
        Parâmetros:
        x0 : chute inicial
        verbose : se True, mostra progresso
        
        Retorna:
        raiz encontrada
        """
        x = float(x0)
        self.history = []
        
        for i in range(self.max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            
            if abs(dfx) < 1e-15:
                raise ValueError(f"Derivada zero em x = {x}")
            
            x_new = x - fx / dfx
            erro = abs(x_new - x)
            
            self.history.append({
                'iter': i + 1,
                'x': x,
                'fx': fx,
                'dfx': dfx,
                'erro': erro
            })
            
            if verbose:
                print(f"Iter {i+1:3d}: x = {x:.10f}, f(x) = {fx:.2e}, erro = {erro:.2e}")
            
            if erro < self.tol or abs(fx) < self.tol:
                self.root_ = x_new
                self.iterations_ = i + 1
                return x_new
            
            x = x_new
        
        self.root_ = x
        self.iterations_ = self.max_iter
        return x
    
    def get_history(self) -> List[dict]:
        """Retorna histórico das iterações."""
        return self.history
    
    def get_stats(self) -> dict:
        """Retorna estatísticas da solução."""
        if self.root_ is None:
            return {}
        
        return {
            'root': self.root_,
            'iterations': self.iterations_,
            'final_residual': abs(self.f(self.root_)),
            'converged': self.iterations_ < self.max_iter,
            'history': self.history
        }
    
    def plot_convergence(self):
        """Plota a convergência do método."""
        if not self.history:
            print("Execute solve primeiro")
            return
        
        plot_convergencia(self.history, self.f)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Executa exemplos
    exemplo_raiz_quadrada()
    exemplo_funcao_transcendental()
    exemplo_raiz_multipla()
    exemplo_sistema_nao_linear()
    exemplo_otimizacao()
    exemplo_comparacao_metodos()
    
    # Demonstração da classe
    print("\n" + "=" * 70)
    print("EXEMPLO 7: Usando a Classe NewtonRaphsonSolver")
    print("=" * 70)
    
    # Exemplo com derivada fornecida
    f = lambda x: x**3 - 2*x - 5
    df = lambda x: 3*x**2 - 2
    
    solver = NewtonRaphsonSolver(f, df, tol=1e-12, max_iter=50)
    raiz = solver.solve(2.0, verbose=True)
    
    print(f"\nResultado final:")
    print(f"Raiz: {raiz:.12f}")
    print(f"f(raiz) = {f(raiz):.2e}")
    print(f"Iterações: {solver.iterations_}")
    
    # Exemplo com derivada numérica
    print("\nCom derivada numérica:")
    solver2 = NewtonRaphsonSolver(f, tol=1e-12, max_iter=50)
    raiz2 = solver2.solve(2.0, verbose=False)
    print(f"Raiz: {raiz2:.12f}")
    print(f"Iterações: {solver2.iterations_}")
    
    # Plot
    solver.plot_convergence()