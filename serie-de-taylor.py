import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union, Optional
from scipy.special import factorial
import sympy as sp
from math import factorial as math_factorial
import warnings

# ============================================================================
# IMPLEMENTAÇÃO BÁSICA DA SÉRIE DE TAYLOR
# ============================================================================

def serie_taylor(f: Callable, 
                 derivadas: List[Callable], 
                 x0: float, 
                 x: Union[float, np.ndarray], 
                 n: int) -> Union[float, np.ndarray]:
    """
    Calcula a série de Taylor usando derivadas fornecidas.
    
    Parâmetros:
    f : função original (para referência)
    derivadas : lista de funções derivadas [f', f'', f''', ...]
    x0 : ponto de expansão
    x : ponto(s) para avaliar
    n : número de termos (ordem n-1)
    
    Retorna:
    aproximação da série de Taylor
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for i in range(n):
        if i == 0:
            termo = f(x0)
        else:
            termo = derivadas[i-1](x0) / math_factorial(i)
        
        resultado += termo * (x - x0)**i
    
    return resultado


def serie_taylor_automatica(f: Callable, 
                            x0: float, 
                            x: Union[float, np.ndarray], 
                            n: int, 
                            h: float = 1e-6) -> Union[float, np.ndarray]:
    """
    Calcula a série de Taylor aproximando as derivadas numericamente.
    
    Parâmetros:
    f : função
    x0 : ponto de expansão
    x : ponto(s) para avaliar
    n : número de termos
    h : passo para diferenças finitas
    
    Retorna:
    aproximação da série de Taylor
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    # Calcula derivadas numericamente
    derivadas = []
    for i in range(n):
        if i == 0:
            derivadas.append(f(x0))
        else:
            # Usa diferença central para derivada
            if i == 1:
                df = (f(x0 + h) - f(x0 - h)) / (2 * h)
            else:
                # Para derivadas de ordem superior, usa diferenças finitas
                # Aproximação simplificada
                df = derivada_numerica(f, x0, i, h)
            derivadas.append(df)
    
    for i in range(n):
        termo = derivadas[i] / math_factorial(i)
        resultado += termo * (x - x0)**i
    
    return resultado


def derivada_numerica(f: Callable, x0: float, ordem: int, h: float = 1e-6) -> float:
    """
    Calcula derivada numérica de ordem superior.
    """
    if ordem == 1:
        return (f(x0 + h) - f(x0 - h)) / (2 * h)
    elif ordem == 2:
        return (f(x0 + h) - 2*f(x0) + f(x0 - h)) / (h**2)
    elif ordem == 3:
        return (f(x0 + 2*h) - 2*f(x0 + h) + 2*f(x0 - h) - f(x0 - 2*h)) / (2 * h**3)
    elif ordem == 4:
        return (f(x0 + 2*h) - 4*f(x0 + h) + 6*f(x0) - 4*f(x0 - h) + f(x0 - 2*h)) / (h**4)
    else:
        # Método genérico usando diferenças finitas
        coeffs = coeficientes_diferencas_finitas(ordem)
        soma = 0
        for i, c in enumerate(coeffs):
            offset = (i - len(coeffs)//2) * h
            soma += c * f(x0 + offset)
        return soma / (h**ordem)


def coeficientes_diferencas_finitas(ordem: int) -> List[float]:
    """
    Retorna coeficientes para diferenças finitas centradas.
    """
    # Coeficientes pré-calculados para ordens comuns
    coeficientes = {
        1: [-0.5, 0, 0.5],
        2: [1, -2, 1],
        3: [-0.5, 1, 0, -1, 0.5],
        4: [1, -4, 6, -4, 1]
    }
    return coeficientes.get(ordem, [1, -1])


# ============================================================================
# SÉRIE DE TAYLOR SIMBÓLICA COM SYMPY
# ============================================================================

def serie_taylor_simbolica(expr: str, 
                           x0: float, 
                           n: int, 
                           var: str = 'x') -> sp.Expr:
    """
    Calcula a série de Taylor simbolicamente usando SymPy.
    
    Parâmetros:
    expr : expressão da função (ex: 'sin(x)', 'exp(x)', 'x**2')
    x0 : ponto de expansão
    n : número de termos
    var : variável independente
    
    Retorna:
    expressão simbólica da série
    """
    x = sp.Symbol(var)
    f = sp.sympify(expr)
    
    # Calcula série de Taylor
    serie = f.series(x, x0, n).removeO()
    
    return serie


def serie_taylor_simbolica_completa(expr: str, 
                                    x0: float, 
                                    n: int, 
                                    var: str = 'x') -> Tuple[sp.Expr, List[sp.Expr]]:
    """
    Calcula série de Taylor e retorna também os termos individuais.
    """
    x = sp.Symbol(var)
    f = sp.sympify(expr)
    
    termos = []
    serie = 0
    
    for i in range(n):
        derivada = sp.diff(f, x, i)
        termo = (derivada.subs(x, x0) / math_factorial(i)) * (x - x0)**i
        termos.append(termo)
        serie += termo
    
    return serie, termos


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE ERROS
# ============================================================================

def erro_taylor(f: Callable, 
                derivada_n1: Callable, 
                x0: float, 
                x: float, 
                n: int) -> float:
    """
    Calcula o erro da série de Taylor usando o resto de Lagrange.
    
    Parâmetros:
    f : função original
    derivada_n1 : derivada de ordem n+1 (para estimar o erro)
    x0 : ponto de expansão
    x : ponto de avaliação
    n : ordem da série
    
    Retorna:
    estimativa do erro
    """
    # Encontra ξ entre x0 e x (usamos o ponto médio como aproximação)
    xi = (x0 + x) / 2
    
    # Calcula derivada de ordem n+1 em ξ
    f_n1_xi = derivada_n1(xi)
    
    # Resto de Lagrange
    erro = abs(f_n1_xi) * abs(x - x0)**(n+1) / math_factorial(n + 1)
    
    return erro


def erro_taylor_pratico(f: Callable, 
                        x0: float, 
                        x: float, 
                        n: int, 
                        h: float = 1e-6) -> float:
    """
    Estima o erro da série de Taylor usando o próximo termo.
    """
    # Aproxima a derivada de ordem n+1
    f_n1 = derivada_numerica(f, x0, n+1, h)
    
    # Estima erro
    erro = abs(f_n1) * abs(x - x0)**(n+1) / math_factorial(n + 1)
    
    return erro


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def plot_serie_taylor(f: Callable, 
                      x0: float, 
                      x_range: Tuple[float, float], 
                      ordens: List[int],
                      num_pontos: int = 500,
                      derivadas: List[Callable] = None):
    """
    Plota a função e suas aproximações por série de Taylor.
    """
    x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
    y_true = f(x_plot)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico principal
    ax1 = axes[0]
    ax1.plot(x_plot, y_true, 'k-', linewidth=2, label='Função original', zorder=10)
    
    cores = plt.cm.viridis(np.linspace(0, 1, len(ordens)))
    
    for ordem, cor in zip(ordens, cores):
        if derivadas:
            y_aprox = serie_taylor(f, derivadas, x0, x_plot, ordem)
        else:
            y_aprox = serie_taylor_automatica(f, x0, x_plot, ordem)
        
        ax1.plot(x_plot, y_aprox, '--', color=cor, linewidth=1.5,
                label=f'Ordem {ordem-1}')
    
    ax1.axvline(x=x0, color='r', linestyle=':', linewidth=2, label=f'x₀ = {x0}')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Série de Taylor - Aproximações', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de erros (log)
    ax2 = axes[1]
    
    for ordem, cor in zip(ordens, cores):
        if derivadas:
            y_aprox = serie_taylor(f, derivadas, x0, x_plot, ordem)
        else:
            y_aprox = serie_taylor_automatica(f, x0, x_plot, ordem)
        
        erro = np.abs(y_aprox - y_true)
        # Evita log de zero
        erro = np.where(erro < 1e-16, 1e-16, erro)
        ax2.semilogy(x_plot, erro, '--', color=cor, linewidth=1.5,
                    label=f'Ordem {ordem-1}')
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Erro Absoluto (log)', fontsize=12)
    ax2.set_title('Erro da Aproximação', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_convergencia_taylor(f: Callable, 
                             x0: float, 
                             x_test: float,
                             max_ordem: int = 20,
                             derivadas: List[Callable] = None):
    """
    Plota a convergência da série de Taylor com o aumento da ordem.
    """
    ordens = range(1, max_ordem + 1)
    erros = []
    aproximacoes = []
    
    y_true = f(x_test)
    
    for n in ordens:
        if derivadas:
            y_aprox = serie_taylor(f, derivadas, x0, x_test, n)
        else:
            y_aprox = serie_taylor_automatica(f, x0, x_test, n)
        
        aproximacoes.append(y_aprox)
        erros.append(abs(y_aprox - y_true))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de convergência
    ax1 = axes[0]
    ax1.semilogy(ordens, erros, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Ordem da Série', fontsize=12)
    ax1.set_ylabel('Erro Absoluto (log)', fontsize=12)
    ax1.set_title(f'Convergência em x = {x_test}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Aproximações
    ax2 = axes[1]
    ax2.plot(ordens, aproximacoes, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=y_true, color='k', linestyle='--', 
                label=f'Valor exato = {y_true:.6f}')
    ax2.set_xlabel('Ordem da Série', fontsize=12)
    ax2.set_ylabel('Aproximação', fontsize=12)
    ax2.set_title(f'Convergência da Aproximação em x = {x_test}', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return erros, aproximacoes


def plot_termos_taylor(f: Callable, 
                       x0: float, 
                       x_range: Tuple[float, float],
                       n_termos: int = 5,
                       derivadas: List[Callable] = None):
    """
    Plota os termos individuais da série de Taylor.
    """
    x_plot = np.linspace(x_range[0], x_range[1], 500)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Termos individuais
    ax1 = axes[0]
    
    if derivadas:
        # Calcula termos
        termos = []
        for i in range(n_termos):
            if i == 0:
                coef = f(x0)
            else:
                coef = derivadas[i-1](x0) / math_factorial(i)
            termo = coef * (x_plot - x0)**i
            termos.append(termo)
            ax1.plot(x_plot, termo, '--', linewidth=1.5, label=f'Termo {i}')
        
        # Soma acumulada
        soma_parcial = np.zeros_like(x_plot)
        for i, termo in enumerate(termos):
            soma_parcial += termo
            if i in [0, 1, 2, n_termos-1]:
                ax1.plot(x_plot, soma_parcial, '-', linewidth=2, 
                        label=f'Soma até ordem {i}')
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=x0, color='r', linestyle=':', linewidth=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Valor', fontsize=12)
    ax1.set_title('Termos da Série de Taylor', fontsize=14)
    ax1.legend(loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Contribuição relativa dos termos
    ax2 = axes[1]
    
    if derivadas:
        # Calcula contribuição em x = x0 + 1
        x_point = x0 + 1
        contribuicoes = []
        for i in range(n_termos):
            if i == 0:
                coef = f(x0)
            else:
                coef = derivadas[i-1](x0) / math_factorial(i)
            termo = coef * (x_point - x0)**i
            contribuicoes.append(abs(termo))
        
        total = sum(contribuicoes)
        contribuicoes_rel = [c/total*100 for c in contribuicoes]
        
        bars = ax2.bar(range(n_termos), contribuicoes_rel, alpha=0.7)
        ax2.set_xlabel('Termo', fontsize=12)
        ax2.set_ylabel('Contribuição Relativa (%)', fontsize=12)
        ax2.set_title(f'Contribuição dos Termos em x = {x_point}', fontsize=14)
        ax2.set_xticks(range(n_termos))
        ax2.grid(True, alpha=0.3)
        
        # Adiciona valores nas barras
        for i, (bar, val) in enumerate(zip(bars, contribuicoes_rel)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SÉRIES DE TAYLOR ESPECIAIS
# ============================================================================

def taylor_exp(x: Union[float, np.ndarray], n: int = 10) -> Union[float, np.ndarray]:
    """
    Série de Taylor para e^x em torno de x=0.
    e^x = Σ x^k / k!
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for k in range(n):
        resultado += x**k / math_factorial(k)
    
    return resultado


def taylor_sin(x: Union[float, np.ndarray], n: int = 10) -> Union[float, np.ndarray]:
    """
    Série de Taylor para sin(x) em torno de x=0.
    sin(x) = Σ (-1)^k x^(2k+1) / (2k+1)!
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for k in range(n):
        termo = ((-1)**k) * x**(2*k + 1) / math_factorial(2*k + 1)
        resultado += termo
    
    return resultado


def taylor_cos(x: Union[float, np.ndarray], n: int = 10) -> Union[float, np.ndarray]:
    """
    Série de Taylor para cos(x) em torno de x=0.
    cos(x) = Σ (-1)^k x^(2k) / (2k)!
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for k in range(n):
        termo = ((-1)**k) * x**(2*k) / math_factorial(2*k)
        resultado += termo
    
    return resultado


def taylor_ln(x: Union[float, np.ndarray], n: int = 10) -> Union[float, np.ndarray]:
    """
    Série de Taylor para ln(1+x) em torno de x=0.
    ln(1+x) = Σ (-1)^(k+1) x^k / k, |x| < 1
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for k in range(1, n + 1):
        termo = ((-1)**(k+1)) * x**k / k
        resultado += termo
    
    return resultado


def taylor_arctan(x: Union[float, np.ndarray], n: int = 10) -> Union[float, np.ndarray]:
    """
    Série de Taylor para arctan(x) em torno de x=0.
    arctan(x) = Σ (-1)^k x^(2k+1) / (2k+1), |x| ≤ 1
    """
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for k in range(n):
        termo = ((-1)**k) * x**(2*k + 1) / (2*k + 1)
        resultado += termo
    
    return resultado


# ============================================================================
# APLICAÇÕES DA SÉRIE DE TAYLOR
# ============================================================================

def aproximar_integral(f: Callable, 
                       a: float, 
                       b: float, 
                       x0: float, 
                       n: int,
                       derivadas: List[Callable]) -> float:
    """
    Aproxima integral usando série de Taylor.
    ∫ f(x) dx ≈ ∫ Σ f^(k)(x0)/k! (x-x0)^k dx
    """
    integral = 0
    
    for i in range(n):
        if i == 0:
            coef = f(x0)
        else:
            coef = derivadas[i-1](x0) / math_factorial(i)
        
        # ∫ (x-x0)^i dx = [(x-x0)^(i+1)]/(i+1) avaliado de a até b
        integral += coef * ((b - x0)**(i+1) - (a - x0)**(i+1)) / (i + 1)
    
    return integral


def resolver_edo_taylor(f: Callable, 
                        y0: float, 
                        x0: float, 
                        h: float, 
                        n_pontos: int,
                        ordem: int) -> np.ndarray:
    """
    Resolve EDO usando série de Taylor (método de Taylor).
    y' = f(x, y), y(x0) = y0
    """
    x = np.zeros(n_pontos)
    y = np.zeros(n_pontos)
    
    x[0] = x0
    y[0] = y0
    
    for i in range(n_pontos - 1):
        # Calcula derivadas de y em x[i]
        derivadas = [y[i]]
        
        # Aproxima derivadas de ordem superior
        for k in range(1, ordem + 1):
            if k == 1:
                derivadas.append(f(x[i], y[i]))
            else:
                # Aproximação simplificada
                derivadas.append(derivadas[k-1] * h)
        
        # Série de Taylor
        y[i+1] = y[i]
        for k in range(1, ordem + 1):
            y[i+1] += derivadas[k] * h**k / math_factorial(k)
        
        x[i+1] = x[i] + h
    
    return x, y


# ============================================================================
# EXEMPLOS E APLICAÇÕES
# ============================================================================

def exemplo_basico():
    """Exemplo básico da série de Taylor."""
    print("=" * 70)
    print("EXEMPLO 1: Série de Taylor - Conceitos Básicos")
    print("=" * 70)
    
    # Função: f(x) = sin(x)
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)
    d2f = lambda x: -np.sin(x)
    d3f = lambda x: -np.cos(x)
    d4f = lambda x: np.sin(x)
    
    derivadas = [df, d2f, d3f, d4f]
    
    x0 = 0
    x_test = 0.5
    
    print(f"Função: f(x) = sin(x)")
    print(f"Ponto de expansão: x₀ = {x0}")
    print(f"Ponto de avaliação: x = {x_test}")
    print(f"Valor exato: sin({x_test}) = {np.sin(x_test):.8f}")
    print()
    
    for n in [1, 2, 3, 4, 5]:
        aprox = serie_taylor(f, derivadas, x0, x_test, n)
        erro = abs(aprox - np.sin(x_test))
        print(f"Ordem {n-1}: P(x) = {aprox:.8f}, Erro = {erro:.2e}")
    
    # Visualização
    plot_serie_taylor(f, x0, (-2*np.pi, 2*np.pi), [2, 3, 4, 5, 6], derivadas)
    plot_convergencia_taylor(f, x0, 0.5, max_ordem=10, derivadas=derivadas)
    
    return derivadas


def exemplo_expansao_centro():
    """Exemplo com diferentes pontos de expansão."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Série de Taylor - Diferentes Pontos de Expansão")
    print("=" * 70)
    
    f = lambda x: np.exp(x)
    df = lambda x: np.exp(x)
    d2f = lambda x: np.exp(x)
    d3f = lambda x: np.exp(x)
    
    derivadas = [df, d2f, d3f, d4f, d5f, d6f]
    
    x0_list = [0, 1, 2]
    x_test = 1.5
    
    print(f"Função: f(x) = e^x")
    print(f"Ponto de avaliação: x = {x_test}")
    print(f"Valor exato: e^{x_test} = {np.exp(x_test):.8f}")
    print()
    
    for x0 in x0_list:
        aprox = serie_taylor(f, derivadas, x0, x_test, 5)
        erro = abs(aprox - np.exp(x_test))
        print(f"Expansão em x₀ = {x0}: P(x) = {aprox:.8f}, Erro = {erro:.2e}")
    
    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x_plot = np.linspace(-1, 3, 500)
    y_true = np.exp(x_plot)
    
    for ax, x0 in zip(axes, [0, 1]):
        y_aprox = serie_taylor(f, derivadas, x0, x_plot, 5)
        
        ax.plot(x_plot, y_true, 'k-', linewidth=2, label='e^x')
        ax.plot(x_plot, y_aprox, 'r--', linewidth=2, label=f'Taylor em x₀={x0}')
        ax.axvline(x=x0, color='b', linestyle=':', label=f'x₀ = {x0}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Expansão em torno de x₀ = {x0}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def exemplo_series_especiais():
    """Demonstra séries de Taylor especiais."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Séries de Taylor Especiais")
    print("=" * 70)
    
    x_test = 0.5
    n_termos = 10
    
    print(f"Avaliação em x = {x_test} com {n_termos} termos:")
    print()
    
    # e^x
    exp_exato = np.exp(x_test)
    exp_aprox = taylor_exp(x_test, n_termos)
    print(f"e^{x_test}:")
    print(f"  Exato: {exp_exato:.10f}")
    print(f"  Taylor: {exp_aprox:.10f}")
    print(f"  Erro: {abs(exp_exato - exp_aprox):.2e}")
    print()
    
    # sin(x)
    sin_exato = np.sin(x_test)
    sin_aprox = taylor_sin(x_test, n_termos)
    print(f"sin({x_test}):")
    print(f"  Exato: {sin_exato:.10f}")
    print(f"  Taylor: {sin_aprox:.10f}")
    print(f"  Erro: {abs(sin_exato - sin_aprox):.2e}")
    print()
    
    # cos(x)
    cos_exato = np.cos(x_test)
    cos_aprox = taylor_cos(x_test, n_termos)
    print(f"cos({x_test}):")
    print(f"  Exato: {cos_exato:.10f}")
    print(f"  Taylor: {cos_aprox:.10f}")
    print(f"  Erro: {abs(cos_exato - cos_aprox):.2e}")
    print()
    
    # ln(1+x)
    ln_exato = np.log(1 + x_test)
    ln_aprox = taylor_ln(x_test, n_termos)
    print(f"ln(1+{x_test}):")
    print(f"  Exato: {ln_exato:.10f}")
    print(f"  Taylor: {ln_aprox:.10f}")
    print(f"  Erro: {abs(ln_exato - ln_aprox):.2e}")
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x_plot = np.linspace(-2, 2, 500)
    
    # e^x
    axes[0, 0].plot(x_plot, np.exp(x_plot), 'k-', linewidth=2, label='e^x')
    axes[0, 0].plot(x_plot, taylor_exp(x_plot, 5), 'r--', label='Ordem 5')
    axes[0, 0].plot(x_plot, taylor_exp(x_plot, 10), 'b--', label='Ordem 10')
    axes[0, 0].set_title('e^x')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # sin(x)
    axes[0, 1].plot(x_plot, np.sin(x_plot), 'k-', linewidth=2, label='sin(x)')
    axes[0, 1].plot(x_plot, taylor_sin(x_plot, 3), 'r--', label='Ordem 3')
    axes[0, 1].plot(x_plot, taylor_sin(x_plot, 7), 'b--', label='Ordem 7')
    axes[0, 1].set_title('sin(x)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # cos(x)
    axes[1, 0].plot(x_plot, np.cos(x_plot), 'k-', linewidth=2, label='cos(x)')
    axes[1, 0].plot(x_plot, taylor_cos(x_plot, 3), 'r--', label='Ordem 3')
    axes[1, 0].plot(x_plot, taylor_cos(x_plot, 7), 'b--', label='Ordem 7')
    axes[1, 0].set_title('cos(x)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ln(1+x)
    x_ln = np.linspace(-0.9, 2, 500)
    axes[1, 1].plot(x_ln, np.log(1 + x_ln), 'k-', linewidth=2, label='ln(1+x)')
    axes[1, 1].plot(x_ln, taylor_ln(x_ln, 5), 'r--', label='Ordem 5')
    axes[1, 1].plot(x_ln, taylor_ln(x_ln, 10), 'b--', label='Ordem 10')
    axes[1, 1].set_title('ln(1+x)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(-0.9, 2)
    
    plt.tight_layout()
    plt.show()


def exemplo_erro_taylor():
    """Demonstra o erro da série de Taylor."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Análise do Erro da Série de Taylor")
    print("=" * 70)
    
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)
    d2f = lambda x: -np.sin(x)
    d3f = lambda x: -np.cos(x)
    d4f = lambda x: np.sin(x)
    d5f = lambda x: np.cos(x)
    
    derivadas = [df, d2f, d3f, d4f, d5f]
    
    x0 = 0
    x_test = 0.5
    
    print(f"Função: sin(x)")
    print(f"Expansão em x₀ = {x0}")
    print(f"Ponto de avaliação: x = {x_test}")
    print()
    
    for n in range(1, 6):
        aprox = serie_taylor(f, derivadas, x0, x_test, n)
        erro_real = abs(aprox - np.sin(x_test))
        
        # Estima erro usando próximo termo
        if n < len(derivadas) + 1:
            termo_seguinte = abs(derivadas[n-1](x0)) * abs(x_test - x0)**n / math_factorial(n)
        else:
            termo_seguinte = 0
        
        print(f"Ordem {n-1}:")
        print(f"  Aproximação: {aprox:.8f}")
        print(f"  Erro real: {erro_real:.2e}")
        print(f"  Estimativa (próximo termo): {termo_seguinte:.2e}")
        print()
    
    # Gráfico do erro em função da distância
    distancias = np.linspace(0, 2, 100)
    erros = []
    
    for d in distancias:
        aprox = serie_taylor(f, derivadas, x0, d, 4)
        erros.append(abs(aprox - np.sin(d)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(distancias, erros, 'b-', linewidth=2)
    plt.xlabel('|x - x₀|', fontsize=12)
    plt.ylabel('Erro', fontsize=12)
    plt.title('Erro da Série de Taylor (ordem 3) em função da distância', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()


def exemplo_aplicacao_fisica():
    """Aplicação em física: pêndulo simples."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Aplicação em Física - Pêndulo Simples")
    print("=" * 70)
    
    # Período do pêndulo: T = 2π √(L/g) * (1 + θ²/16 + ...)
    L = 1.0  # comprimento (m)
    g = 9.81  # gravidade (m/s²)
    
    T0 = 2 * np.pi * np.sqrt(L / g)  # Período para pequenas oscilações
    
    def periodo_taylor(theta_max_rad):
        """Período usando série de Taylor."""
        # Série: T = T0 * (1 + θ²/16 + 11θ⁴/3072 + ...)
        theta2 = theta_max_rad**2
        theta4 = theta_max_rad**4
        
        return T0 * (1 + theta2/16 + 11*theta4/3072)
    
    def periodo_exato(theta_max_rad):
        """Período exato (integral elíptica)."""
        from scipy.special import ellipk
        k = np.sin(theta_max_rad / 2)
        return 4 * np.sqrt(L/g) * ellipk(k**2)
    
    angulos_graus = np.linspace(0, 90, 10)
    angulos_rad = np.deg2rad(angulos_graus)
    
    print("Ângulo (°) | Período Aprox (s) | Período Exato (s) | Erro (%)")
    print("-" * 60)
    
    for theta_deg, theta_rad in zip(angulos_graus, angulos_rad):
        T_aprox = periodo_taylor(theta_rad)
        T_exato = periodo_exato(theta_rad)
        erro_percent = abs(T_aprox - T_exato) / T_exato * 100
        
        print(f"{theta_deg:8.0f}   | {T_aprox:14.4f}   | {T_exato:13.4f}   | {erro_percent:6.2f}")
    
    # Visualização
    theta_range = np.deg2rad(np.linspace(0, 90, 100))
    T_aprox = periodo_taylor(theta_range)
    T_exato = periodo_exato(theta_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(theta_range), T_aprox, 'b-', linewidth=2, label='Aproximação (Taylor)')
    plt.plot(np.rad2deg(theta_range), T_exato, 'r--', linewidth=2, label='Exato')
    plt.xlabel('Ângulo máximo (graus)', fontsize=12)
    plt.ylabel('Período (s)', fontsize=12)
    plt.title('Período do Pêndulo Simples', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def exemplo_simbolico():
    """Exemplo com cálculo simbólico usando SymPy."""
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Série de Taylor Simbólica com SymPy")
    print("=" * 70)
    
    try:
        # Funções para expansão
        funcoes = [
            ('sin(x)', 0),
            ('cos(x)', 0),
            ('exp(x)', 0),
            ('ln(1+x)', 0),
            ('1/(1-x)', 0),
            ('sqrt(1+x)', 0)
        ]
        
        for expr, x0 in funcoes:
            print(f"\n{expr} em torno de x = {x0}:")
            serie = serie_taylor_simbolica(expr, x0, 6)
            print(f"  {serie}")
        
        # Série com termos individuais
        expr = 'sin(x)'
        x0 = 0
        serie, termos = serie_taylor_simbolica_completa(expr, x0, 6)
        
        print(f"\nTermos individuais para {expr}:")
        for i, termo in enumerate(termos):
            if termo != 0:
                print(f"  Termo {i}: {termo}")
        
        print(f"\nSérie completa: {serie}")
        
    except ImportError:
        print("SymPy não está instalado. Instale com: pip install sympy")


def exemplo_edo_taylor():
    """Resolve EDO usando série de Taylor."""
    print("\n" + "=" * 70)
    print("EXEMPLO 7: Resolução de EDO com Série de Taylor")
    print("=" * 70)
    
    # EDO: y' = -2xy, y(0) = 1
    # Solução exata: y = exp(-x²)
    
    def f_edo(x, y):
        return -2 * x * y
    
    x0 = 0
    y0 = 1
    h = 0.1
    n_pontos = 50
    ordem = 4
    
    x, y_aprox = resolver_edo_taylor(f_edo, y0, x0, h, n_pontos, ordem)
    y_exato = np.exp(-x**2)
    
    print(f"EDO: y' = -2xy, y(0) = 1")
    print(f"Solução exata: y = e^(-x²)")
    print(f"Passo: h = {h}")
    print(f"Ordem da série: {ordem}")
    print()
    
    print(f"{'x':^10} {'Aproximado':^12} {'Exato':^12} {'Erro':^12}")
    print("-" * 50)
    
    for i in range(0, n_pontos, 10):
        print(f"{x[i]:^10.2f} {y_aprox[i]:^12.6f} {y_exato[i]:^12.6f} {abs(y_aprox[i]-y_exato[i]):^12.2e}")
    
    # Visualização
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_exato, 'k-', linewidth=2, label='Exato')
    plt.plot(x, y_aprox, 'ro--', linewidth=1.5, markersize=3, label='Taylor (ordem 4)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solução da EDO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    erro = np.abs(y_aprox - y_exato)
    plt.semilogy(x, erro, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Erro (log)')
    plt.title('Erro da Aproximação')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# CLASSE PARA SÉRIE DE TAYLOR
# ============================================================================

class TaylorSeries:
    """
    Classe para manipulação de séries de Taylor.
    """
    
    def __init__(self, f: Callable, x0: float, derivadas: List[Callable] = None):
        """
        Parâmetros:
        f : função
        x0 : ponto de expansão
        derivadas : lista de derivadas (opcional)
        """
        self.f = f
        self.x0 = x0
        self.derivadas = derivadas if derivadas else []
        self._coeficientes = None
        self._max_ordem = len(derivadas) if derivadas else 0
    
    def coeficiente(self, n: int) -> float:
        """Retorna o coeficiente do termo de ordem n."""
        if n == 0:
            return self.f(self.x0)
        
        if n - 1 < len(self.derivadas):
            return self.derivadas[n-1](self.x0) / math_factorial(n)
        else:
            # Calcula derivada numericamente
            deriv = derivada_numerica(self.f, self.x0, n)
            return deriv / math_factorial(n)
    
    def __call__(self, x: Union[float, np.ndarray], n: int) -> Union[float, np.ndarray]:
        """Avalia a série de Taylor até ordem n."""
        x_arr = np.array(x, dtype=float)
        resultado = np.zeros_like(x_arr)
        
        for i in range(n):
            coef = self.coeficiente(i)
            resultado += coef * (x_arr - self.x0)**i
        
        return resultado
    
    def get_serie(self, n: int) -> str:
        """Retorna representação simbólica da série."""
        termos = []
        for i in range(n):
            coef = self.coeficiente(i)
            if abs(coef) < 1e-12:
                continue
            
            if i == 0:
                termo = f"{coef:.6f}"
            else:
                sinal = "+" if coef > 0 else "-"
                termo = f"{sinal} {abs(coef):.6f} (x - {self.x0})^{i}"
            termos.append(termo)
        
        return "P(x) = " + " ".join(termos)
    
    def plot(self, x_range: Tuple[float, float], ordens: List[int], num_pontos: int = 500):
        """Plota a função e suas aproximações."""
        plot_serie_taylor(self.f, self.x0, x_range, ordens, num_pontos, self.derivadas)
    
    def convergencia(self, x_test: float, max_ordem: int = 20):
        """Estuda a convergência da série."""
        return plot_convergencia_taylor(self.f, self.x0, x_test, max_ordem, self.derivadas)


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Executa exemplos
    exemplo_basico()
    exemplo_expansao_centro()
    exemplo_series_especiais()
    exemplo_erro_taylor()
    exemplo_aplicacao_fisica()
    exemplo_simbolico()
    exemplo_edo_taylor()
    
    # Demonstração da classe
    print("\n" + "=" * 70)
    print("EXEMPLO 8: Usando a Classe TaylorSeries")
    print("=" * 70)
    
    # Função: f(x) = ln(1+x)
    f = lambda x: np.log(1 + x)
    df = lambda x: 1/(1 + x)
    d2f = lambda x: -1/(1 + x)**2
    d3f = lambda x: 2/(1 + x)**3
    d4f = lambda x: -6/(1 + x)**4
    
    derivadas = [df, d2f, d3f, d4f]
    
    # Cria série
    taylor = TaylorSeries(f, 0, derivadas)
    
    print(f"Função: ln(1+x)")
    print(f"Ponto de expansão: x₀ = 0")
    print(f"\nSérie até ordem 5:")
    print(taylor.get_serie(5))
    
    # Teste
    x_test = 0.5
    print(f"\nAvaliação em x = {x_test}:")
    print(f"  Valor exato: {np.log(1 + x_test):.8f}")
    
    for n in [2, 3, 4, 5]:
        aprox = taylor(x_test, n)
        erro = abs(aprox - np.log(1 + x_test))
        print(f"  Ordem {n-1}: {aprox:.8f}, erro = {erro:.2e}")
    
    # Plot
    taylor.plot(x_range=(-0.5, 1.5), ordens=[2, 3, 4, 5])
    taylor.convergencia(x_test=0.5, max_ordem=10)
    
    print("\n" + "=" * 70)
    print("RESUMO DAS APLICAÇÕES DA SÉRIE DE TAYLOR")
    print("=" * 70)
    print("""
    1. Aproximação de funções complexas
    2. Cálculo de limites (regra de L'Hôpital)
    3. Integração numérica
    4. Resolução de equações diferenciais
    5. Linearização de sistemas não lineares
    6. Análise de erros em métodos numéricos
    7. Física (pêndulo, relatividade, etc.)
    8. Engenharia (séries de potência)
    """)