import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union

# ============================================================================
# IMPLEMENTAÇÃO BÁSICA DA INTERPOLAÇÃO DE LAGRANGE
# ============================================================================

def polinomio_lagrange(x: Union[float, np.ndarray], 
                       x_i: List[float], 
                       y_i: List[float]) -> Union[float, np.ndarray]:
    """
    Avalia o polinômio interpolador de Lagrange nos pontos x.
    
    Parâmetros:
    x : ponto(s) onde avaliar o polinômio
    x_i : lista de pontos x conhecidos (nós de interpolação)
    y_i : lista de valores y conhecidos (f(x_i))
    
    Retorna:
    valor do polinômio interpolador em x
    """
    n = len(x_i)
    resultado = 0.0
    
    # Converte para array se for escalar
    x = np.array(x, dtype=float)
    resultado = np.zeros_like(x)
    
    for i in range(n):
        # Calcula o polinômio base L_i(x)
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i = L_i * (x - x_i[j]) / (x_i[i] - x_i[j])
        
        # Adiciona a contribuição de L_i * y_i
        resultado += L_i * y_i[i]
    
    return resultado


def lagrange_coeficientes(x_i: List[float], y_i: List[float]) -> List[np.poly1d]:
    """
    Retorna os polinômios base de Lagrange como objetos poly1d.
    
    Parâmetros:
    x_i : lista de pontos x
    y_i : lista de valores y
    
    Retorna:
    lista de polinômios base L_i(x)
    """
    n = len(x_i)
    polinomios_base = []
    
    for i in range(n):
        # Cria polinômio L_i começando com 1
        p = np.poly1d([1])
        
        for j in range(n):
            if j != i:
                # Multiplica por (x - x_j)/(x_i - x_j)
                num = np.poly1d([1, -x_i[j]])
                den = x_i[i] - x_i[j]
                p = p * (num / den)
        
        polinomios_base.append(p * y_i[i])
    
    return polinomios_base


def polinomio_lagrange_completo(x_i: List[float], y_i: List[float]) -> np.poly1d:
    """
    Retorna o polinômio interpolador completo como objeto poly1d.
    
    Parâmetros:
    x_i : lista de pontos x
    y_i : lista de valores y
    
    Retorna:
    polinômio interpolador P(x)
    """
    n = len(x_i)
    polinomio = np.poly1d([0])
    
    for i in range(n):
        # Calcula L_i(x)
        L_i = np.poly1d([1])
        for j in range(n):
            if j != i:
                num = np.poly1d([1, -x_i[j]])
                den = x_i[i] - x_i[j]
                L_i = L_i * (num / den)
        
        polinomio = polinomio + L_i * y_i[i]
    
    return polinomio


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE ERROS
# ============================================================================

def erro_interpolacao(x: Union[float, np.ndarray], 
                      x_i: List[float], 
                      f: Callable, 
                      f_derivada_max: float = None) -> float:
    """
    Estima o erro da interpolação de Lagrange.
    
    Teorema: |f(x) - P(x)| ≤ |ω(x)| * max|f^(n+1)(ξ)| / (n+1)!
    onde ω(x) = ∏(x - x_i)
    
    Parâmetros:
    x : ponto onde avaliar o erro
    x_i : nós de interpolação
    f : função original
    f_derivada_max : valor máximo da derivada de ordem n+1 (se None, estima numericamente)
    
    Retorna:
    estimativa do erro
    """
    n = len(x_i)
    x = np.array(x, dtype=float)
    
    # Calcula ω(x) = ∏(x - x_i)
    omega = np.ones_like(x)
    for xi in x_i:
        omega = omega * (x - xi)
    
    if f_derivada_max is None:
        # Estimativa grosseira usando diferenças finitas
        # Em geral, usa-se um limitante superior conhecido
        f_derivada_max = 1.0  # Valor padrão (ajustar conforme necessidade)
    
    # Erro = |ω(x)| * M / (n+1)!
    erro = np.abs(omega) * f_derivada_max / np.math.factorial(n + 1)
    
    return erro if isinstance(erro, float) else erro[0] if len(erro) == 1 else erro


def comparar_interpolacao(x_i: List[float], 
                          y_i: List[float], 
                          f_original: Callable,
                          x_test: np.ndarray = None) -> dict:
    """
    Compara o polinômio interpolador com a função original.
    
    Parâmetros:
    x_i : nós de interpolação
    y_i : valores nos nós
    f_original : função original (para comparação)
    x_test : pontos de teste (se None, usa pontos igualmente espaçados)
    
    Retorna:
    dicionário com métricas de erro
    """
    if x_test is None:
        x_test = np.linspace(min(x_i), max(x_i), 1000)
    
    # Avalia polinômio interpolador
    y_interp = polinomio_lagrange(x_test, x_i, y_i)
    
    # Avalia função original
    y_original = f_original(x_test)
    
    # Calcula erros
    erros_abs = np.abs(y_interp - y_original)
    erros_rel = erros_abs / (np.abs(y_original) + 1e-10)
    
    resultados = {
        'x_test': x_test,
        'y_original': y_original,
        'y_interp': y_interp,
        'erro_maximo': np.max(erros_abs),
        'erro_medio': np.mean(erros_abs),
        'erro_rmse': np.sqrt(np.mean(erros_abs**2)),
        'erro_relativo_maximo': np.max(erros_rel),
        'erro_relativo_medio': np.mean(erros_rel),
        'max_diferenca_relativa': np.max(erros_rel) * 100,
        'pontos_max_erro': x_test[np.argmax(erros_abs)]
    }
    
    return resultados


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def plot_interpolacao(x_i: List[float], 
                      y_i: List[float], 
                      f_original: Callable = None,
                      x_range: Tuple[float, float] = None,
                      num_pontos: int = 1000,
                      mostrar_erro: bool = True):
    """
    Plota a interpolação de Lagrange e a função original.
    
    Parâmetros:
    x_i : nós de interpolação
    y_i : valores nos nós
    f_original : função original (opcional)
    x_range : intervalo para plotagem (min, max)
    num_pontos : número de pontos para a curva suave
    mostrar_erro : se True, plota também o erro
    """
    if x_range is None:
        x_range = (min(x_i), max(x_i))
    
    x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
    y_interp = polinomio_lagrange(x_plot, x_i, y_i)
    
    fig, axes = plt.subplots(1, 2 if mostrar_erro and f_original else 1, 
                             figsize=(14, 5))
    
    if not mostrar_erro or f_original is None:
        axes = [axes]
    
    # Gráfico principal
    ax = axes[0]
    ax.plot(x_plot, y_interp, 'b-', linewidth=2, label='Polinômio de Lagrange')
    ax.plot(x_i, y_i, 'ro', markersize=8, label='Pontos de interpolação')
    
    if f_original:
        y_original = f_original(x_plot)
        ax.plot(x_plot, y_original, 'g--', linewidth=1.5, label='Função original', alpha=0.7)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Interpolação de Lagrange', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gráfico de erro
    if mostrar_erro and f_original:
        ax2 = axes[1]
        erro = np.abs(y_interp - f_original(x_plot))
        ax2.semilogy(x_plot, erro, 'r-', linewidth=2)
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Erro Absoluto (log)', fontsize=12)
        ax2.set_title('Erro da Interpolação', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Marca os pontos de interpolação (erro zero)
        erro_nos = np.abs(y_i - f_original(x_i)) if f_original else np.zeros_like(x_i)
        ax2.plot(x_i, erro_nos, 'bo', markersize=6, label='Nós (erro zero)')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_bases_lagrange(x_i: List[float], 
                        x_range: Tuple[float, float] = None,
                        num_pontos: int = 500):
    """
    Plota os polinômios base de Lagrange L_i(x).
    
    Parâmetros:
    x_i : nós de interpolação
    x_range : intervalo para plotagem
    num_pontos : número de pontos para as curvas
    """
    if x_range is None:
        x_range = (min(x_i) - 0.5, max(x_i) + 0.5)
    
    x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
    n = len(x_i)
    
    # Calcula bases
    bases = []
    for i in range(n):
        L_i = np.ones_like(x_plot)
        for j in range(n):
            if j != i:
                L_i = L_i * (x_plot - x_i[j]) / (x_i[i] - x_i[j])
        bases.append(L_i)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cores = plt.cm.tab10(np.linspace(0, 1, n))
    
    for i in range(n):
        ax.plot(x_plot, bases[i], color=cores[i], linewidth=2, label=f'L_{i}(x)')
        ax.plot(x_i[i], 1.0, 'o', color=cores[i], markersize=8)
        ax.plot(x_i[i], 0.0, 'x', color=cores[i], markersize=8)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('L_i(x)', fontsize=12)
    ax.set_title('Polinômios Base de Lagrange', fontsize=14)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXEMPLOS E APLICAÇÕES
# ============================================================================

def exemplo_funcao_seno():
    """Exemplo com função seno."""
    print("=" * 70)
    print("EXEMPLO 1: Interpolação da função f(x) = sin(x)")
    print("=" * 70)
    
    # Nós de interpolação
    x_i = np.linspace(0, np.pi, 5)
    y_i = np.sin(x_i)
    
    print(f"Nós de interpolação: {x_i}")
    print(f"Valores: {y_i}")
    
    # Obtém polinômio
    P = polinomio_lagrange_completo(x_i, y_i)
    print(f"\nPolinômio interpolador:\n{P}")
    
    # Testa em alguns pontos
    x_test = [0.5, 1.0, 1.5, 2.0, 2.5]
    for x in x_test:
        valor_exato = np.sin(x)
        valor_interp = polinomio_lagrange(x, x_i, y_i)
        erro = abs(valor_interp - valor_exato)
        print(f"x = {x:.2f}: sin(x) = {valor_exato:.6f}, P(x) = {valor_interp:.6f}, erro = {erro:.2e}")
    
    # Comparação detalhada
    resultados = comparar_interpolacao(x_i, y_i, np.sin)
    print(f"\nMétricas de erro:")
    print(f"  Erro máximo: {resultados['erro_maximo']:.2e}")
    print(f"  Erro médio: {resultados['erro_medio']:.2e}")
    print(f"  RMSE: {resultados['erro_rmse']:.2e}")
    print(f"  Erro relativo máximo: {resultados['erro_relativo_maximo']:.2e}")
    
    # Plot
    plot_interpolacao(x_i, y_i, np.sin, mostrar_erro=True)


def exemplo_funcao_exponencial():
    """Exemplo com função exponencial."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Interpolação da função f(x) = e^x")
    print("=" * 70)
    
    # Nós de interpolação
    x_i = np.linspace(0, 2, 4)
    y_i = np.exp(x_i)
    
    print(f"Nós: {x_i}")
    
    # Teste com diferentes números de pontos
    for n_pontos in [3, 5, 7]:
        x_i_n = np.linspace(0, 2, n_pontos)
        y_i_n = np.exp(x_i_n)
        
        resultados = comparar_interpolacao(x_i_n, y_i_n, np.exp)
        print(f"\nCom {n_pontos} pontos:")
        print(f"  Erro máximo: {resultados['erro_maximo']:.2e}")
        print(f"  RMSE: {resultados['erro_rmse']:.2e}")
    
    # Plot
    plot_interpolacao(x_i, y_i, np.exp, mostrar_erro=True)


def exemplo_fenomeno_runge():
    """Demonstra o fenômeno de Runge."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Fenômeno de Runge - f(x) = 1/(1+25x²)")
    print("=" * 70)
    
    def runge(x):
        return 1 / (1 + 25 * x**2)
    
    # Interpolação com pontos igualmente espaçados
    n_pontos = 10
    x_i_eq = np.linspace(-1, 1, n_pontos)
    y_i_eq = runge(x_i_eq)
    
    # Interpolação com pontos de Chebyshev (melhor distribuição)
    x_i_cheb = np.cos(np.linspace(0, np.pi, n_pontos))  # Pontos de Chebyshev em [-1,1]
    y_i_cheb = runge(x_i_cheb)
    
    # Compara as duas abordagens
    resultados_eq = comparar_interpolacao(x_i_eq, y_i_eq, runge)
    resultados_cheb = comparar_interpolacao(x_i_cheb, y_i_cheb, runge)
    
    print(f"Com {n_pontos} pontos igualmente espaçados:")
    print(f"  Erro máximo: {resultados_eq['erro_maximo']:.2e}")
    print(f"  Erro no centro: {np.abs(polinomio_lagrange(0, x_i_eq, y_i_eq) - runge(0)):.2e}")
    
    print(f"\nCom {n_pontos} pontos de Chebyshev:")
    print(f"  Erro máximo: {resultados_cheb['erro_maximo']:.2e}")
    print(f"  Erro no centro: {np.abs(polinomio_lagrange(0, x_i_cheb, y_i_cheb) - runge(0)):.2e}")
    
    # Plot comparativo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_plot = np.linspace(-1, 1, 1000)
    y_true = runge(x_plot)
    
    for ax, x_i, y_i, titulo in zip(axes, [x_i_eq, x_i_cheb], 
                                     [y_i_eq, y_i_cheb],
                                     ['Pontos Equiespaçados', 'Pontos de Chebyshev']):
        y_interp = polinomio_lagrange(x_plot, x_i, y_i)
        
        ax.plot(x_plot, y_true, 'g-', linewidth=2, label='Função original')
        ax.plot(x_plot, y_interp, 'r--', linewidth=1.5, label='Interpolação')
        ax.plot(x_i, y_i, 'bo', markersize=6, label='Nós')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'{titulo} (n = {len(x_i)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def exemplo_dados_experimentais():
    """Exemplo com dados experimentais."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Interpolação de Dados Experimentais")
    print("=" * 70)
    
    # Dados de um experimento (temperatura vs pressão)
    temperatura = np.array([0, 20, 40, 60, 80, 100])
    pressao = np.array([1.0, 1.2, 1.5, 1.9, 2.4, 3.0])
    
    print("Dados experimentais:")
    print(f"Temperatura (°C): {temperatura}")
    print(f"Pressão (atm): {pressao}")
    
    # Interpola para novos pontos
    temp_interp = np.array([10, 30, 50, 70, 90])
    press_interp = polinomio_lagrange(temp_interp, temperatura, pressao)
    
    print("\nValores interpolados:")
    for t, p in zip(temp_interp, press_interp):
        print(f"T = {t:3.0f}°C → P = {p:.3f} atm")
    
    # Plot
    x_plot = np.linspace(0, 100, 200)
    y_plot = polinomio_lagrange(x_plot, temperatura, pressao)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio interpolador')
    plt.plot(temperatura, pressao, 'ro', markersize=8, label='Dados experimentais')
    plt.plot(temp_interp, press_interp, 'gs', markersize=8, label='Pontos interpolados')
    plt.xlabel('Temperatura (°C)', fontsize=12)
    plt.ylabel('Pressão (atm)', fontsize=12)
    plt.title('Interpolação de Dados de Pressão vs Temperatura', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def exemplo_bases_lagrange():
    """Demonstra os polinômios base de Lagrange."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Polinômios Base de Lagrange")
    print("=" * 70)
    
    # Exemplo com 4 pontos
    x_i = [0, 1, 2, 3]
    y_i = [1, 2, 1, 0]  # Valores arbitrários
    
    # Obtém bases
    bases = lagrange_coeficientes(x_i, y_i)
    
    print(f"Para os pontos x = {x_i}")
    print("Polinômios base L_i(x) * y_i:")
    for i, base in enumerate(bases):
        print(f"  Termo {i}: {base}")
    
    # Soma dos termos = polinômio completo
    P = sum(bases)
    print(f"\nPolinômio completo: {P}")
    
    # Verifica nos pontos
    for x in x_i:
        print(f"P({x}) = {P(x)} (esperado: {y_i[x_i.index(x)]})")
    
    # Plota bases
    plot_bases_lagrange(x_i, x_range=(-0.5, 3.5))


# ============================================================================
# CLASSE PARA INTERPOLAÇÃO DE LAGRANGE
# ============================================================================

class LagrangeInterpolator:
    """
    Classe para interpolação de Lagrange com caching dos polinômios base.
    """
    
    def __init__(self, x_i: List[float], y_i: List[float]):
        """
        Inicializa o interpolador.
        
        Parâmetros:
        x_i : lista de pontos x
        y_i : lista de valores y
        """
        self.x_i = np.array(x_i, dtype=float)
        self.y_i = np.array(y_i, dtype=float)
        self.n = len(x_i)
        
        # Pré-computa os polinômios base
        self.bases = self._compute_bases()
        
    def _compute_bases(self) -> List[np.poly1d]:
        """Pré-computa os polinômios base L_i(x)."""
        bases = []
        
        for i in range(self.n):
            L_i = np.poly1d([1])
            for j in range(self.n):
                if j != i:
                    num = np.poly1d([1, -self.x_i[j]])
                    den = self.x_i[i] - self.x_i[j]
                    L_i = L_i * (num / den)
            bases.append(L_i)
        
        return bases
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Avalia o polinômio interpolador em x.
        
        Parâmetros:
        x : ponto(s) para avaliação
        
        Retorna:
        valor(es) do polinômio
        """
        x = np.array(x, dtype=float)
        resultado = np.zeros_like(x)
        
        for i in range(self.n):
            resultado += self.bases[i](x) * self.y_i[i]
        
        return resultado if len(resultado) > 1 else resultado[0]
    
    def get_polynomial(self) -> np.poly1d:
        """Retorna o polinômio completo."""
        P = np.poly1d([0])
        for i in range(self.n):
            P = P + self.bases[i] * self.y_i[i]
        return P
    
    def plot(self, x_range: Tuple[float, float] = None, num_pontos: int = 500):
        """Plota o polinômio interpolador."""
        if x_range is None:
            x_range = (min(self.x_i) - 0.5, max(self.x_i) + 0.5)
        
        x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
        y_plot = self(x_plot)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio interpolador')
        plt.plot(self.x_i, self.y_i, 'ro', markersize=8, label='Pontos de interpolação')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('Interpolação de Lagrange', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# ============================================================================
# TESTES E DEMONSTRAÇÕES
# ============================================================================

if __name__ == "__main__":
    
    # Executa exemplos
    exemplo_funcao_seno()
    exemplo_funcao_exponencial()
    exemplo_fenomeno_runge()
    exemplo_dados_experimentais()
    exemplo_bases_lagrange()
    
    # Demonstração da classe
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Usando a Classe LagrangeInterpolator")
    print("=" * 70)
    
    # Dados
    x_i = [0, 1, 2, 3, 4]
    y_i = [0, 1, 4, 9, 16]  # f(x) = x²
    
    # Cria interpolador
    interpolador = LagrangeInterpolator(x_i, y_i)
    
    # Testa
    x_test = 2.5
    print(f"Interpolação em x = {x_test}: {interpolador(x_test):.6f}")
    print(f"Valor exato (x²): {x_test**2:.6f}")
    
    # Obtém polinômio
    P = interpolador.get_polynomial()
    print(f"\nPolinômio encontrado: {P}")
    
    # Verifica
    print("\nVerificação nos nós:")
    for x, y in zip(x_i, y_i):
        print(f"P({x}) = {P(x):.6f} (esperado: {y})")
    
    # Plot
    interpolador.plot(x_range=(-0.5, 4.5))