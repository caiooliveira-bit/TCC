import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union, Optional
from scipy import interpolate

# ============================================================================
# IMPLEMENTAÇÃO BÁSICA DAS DIFERENÇAS DIVIDIDAS DE NEWTON
# ============================================================================

def diferencas_divididas(x: List[float], y: List[float]) -> np.ndarray:
    """
    Calcula as diferenças divididas de Newton.
    
    Parâmetros:
    x : lista de pontos x (nós de interpolação)
    y : lista de valores f(x)
    
    Retorna:
    coeficientes : array com os coeficientes das diferenças divididas
    """
    n = len(x)
    # Cria tabela de diferenças divididas
    dd = np.zeros((n, n))
    dd[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x[i+j] - x[i])
    
    # Retorna primeira linha (coeficientes)
    return dd[0, :]


def polinomio_newton(x: Union[float, np.ndarray], 
                     x_i: List[float], 
                     coeficientes: np.ndarray) -> Union[float, np.ndarray]:
    """
    Avalia o polinômio de Newton nos pontos x.
    
    Parâmetros:
    x : ponto(s) para avaliação
    x_i : nós de interpolação
    coeficientes : coeficientes das diferenças divididas
    
    Retorna:
    valor do polinômio em x
    """
    x = np.array(x, dtype=float)
    n = len(coeficientes)
    resultado = np.zeros_like(x)
    
    # Avalia usando o esquema de Horner aninhado
    for i in range(n):
        termo = coeficientes[i]
        for j in range(i):
            termo = termo * (x - x_i[j])
        resultado += termo
    
    return resultado


def polinomio_newton_horner(x: Union[float, np.ndarray], 
                            x_i: List[float], 
                            coeficientes: np.ndarray) -> Union[float, np.ndarray]:
    """
    Avalia o polinômio de Newton usando o algoritmo de Horner (mais eficiente).
    """
    x = np.array(x, dtype=float)
    n = len(coeficientes)
    resultado = np.zeros_like(x)
    
    # Para cada ponto x
    for idx, x_val in enumerate(x):
        # Algoritmo de Horner aninhado
        p = coeficientes[-1]
        for i in range(n-2, -1, -1):
            p = p * (x_val - x_i[i]) + coeficientes[i]
        resultado[idx] = p
    
    return resultado if len(resultado) > 1 else resultado[0]


def newton_coeficientes_simbolicos(x_i: List[float], y_i: List[float]) -> str:
    """
    Retorna a representação simbólica do polinômio de Newton.
    """
    coef = diferencas_divididas(x_i, y_i)
    n = len(coef)
    
    termos = []
    for i in range(n):
        if abs(coef[i]) < 1e-12:
            continue
        
        if i == 0:
            termo = f"{coef[i]:.6f}"
        else:
            termo = f"{coef[i]:+.6f}"
            for j in range(i):
                termo += f"(x - {x_i[j]:.4f})"
        
        termos.append(termo)
    
    return "P(x) = " + " ".join(termos)


# ============================================================================
# INTERPOLAÇÃO DE NEWTON COM DIFERENTES ESTRATÉGIAS
# ============================================================================

def interpolacao_newton_avancada(x: List[float], y: List[float], 
                                 x_interp: Union[float, np.ndarray],
                                 pontos: int = None) -> Union[float, np.ndarray]:
    """
    Interpolação de Newton com escolha automática da ordem.
    
    Parâmetros:
    x : pontos conhecidos
    y : valores conhecidos
    x_interp : ponto(s) para interpolar
    pontos : número de pontos a usar (None = usa todos)
    """
    x = np.array(x)
    y = np.array(y)
    x_interp = np.array(x_interp)
    
    if pontos is not None:
        # Ordena por distância ao ponto de interpolação
        # (útil para interpolar em um ponto específico)
        distancias = np.abs(x - x_interp[0])
        idx = np.argsort(distancias)[:pontos]
        x = x[idx]
        y = y[idx]
        # Reordena por x para estabilidade
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    
    coef = diferencas_divididas(x, y)
    return polinomio_newton(x_interp, x, coef)


def newton_gregory_forward(x: List[float], y: List[float], 
                           x_interp: float) -> float:
    """
    Fórmula de Newton-Gregory para pontos igualmente espaçados (diferenças finitas progressivas).
    
    Parâmetros:
    x : pontos igualmente espaçados
    y : valores
    x_interp : ponto para interpolar (deve estar no início do intervalo)
    """
    x = np.array(x)
    y = np.array(y)
    
    h = x[1] - x[0]  # Passo
    s = (x_interp - x[0]) / h
    
    n = len(y)
    
    # Calcula diferenças finitas
    diff = np.zeros((n, n))
    diff[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            diff[i, j] = diff[i+1, j-1] - diff[i, j-1]
    
    # Avalia usando fórmula de Gregory-Newton
    resultado = diff[0, 0]
    termo = 1
    
    for i in range(1, n):
        termo = termo * (s - i + 1) / i
        resultado += termo * diff[0, i]
    
    return resultado


def newton_gregory_backward(x: List[float], y: List[float], 
                            x_interp: float) -> float:
    """
    Fórmula de Newton-Gregory para pontos igualmente espaçados (diferenças finitas regressivas).
    
    Parâmetros:
    x : pontos igualmente espaçados
    y : valores
    x_interp : ponto para interpolar (deve estar no final do intervalo)
    """
    x = np.array(x)
    y = np.array(y)
    
    h = x[1] - x[0]
    s = (x_interp - x[-1]) / h
    
    n = len(y)
    
    # Calcula diferenças finitas
    diff = np.zeros((n, n))
    diff[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            diff[i, j] = diff[i+1, j-1] - diff[i, j-1]
    
    # Avalia usando fórmula regressiva
    resultado = diff[-1, 0]
    termo = 1
    
    for i in range(1, n):
        termo = termo * (s + i - 1) / i
        resultado += termo * diff[-i-1, i]
    
    return resultado


# ============================================================================
# FUNÇÕES PARA ADIÇÃO DINÂMICA DE PONTOS (FÓRMULA DE NEWTON)
# ============================================================================

class NewtonInterpolator:
    """
    Classe para interpolação de Newton com adição dinâmica de pontos.
    """
    
    def __init__(self):
        self.x_points = []
        self.y_points = []
        self.coeficientes = None
        self.tabela_dd = None
    
    def add_point(self, x: float, y: float):
        """
        Adiciona um novo ponto de interpolação.
        """
        self.x_points.append(x)
        self.y_points.append(y)
        self._update_coeficientes()
    
    def _update_coeficientes(self):
        """
        Atualiza os coeficientes das diferenças divididas.
        """
        n = len(self.x_points)
        
        if n == 1:
            self.coeficientes = np.array([self.y_points[0]])
            self.tabela_dd = np.array([[self.y_points[0]]])
            return
        
        # Cria nova tabela de diferenças
        nova_tabela = np.zeros((n, n))
        nova_tabela[:, 0] = self.y_points
        
        # Copia parte da tabela antiga
        if self.tabela_dd is not None:
            m = n - 1
            nova_tabela[:m, :m] = self.tabela_dd
        
        # Calcula novas diferenças
        for j in range(1, n):
            i = n - j - 1
            nova_tabela[i, j] = (nova_tabela[i+1, j-1] - nova_tabela[i, j-1]) / \
                                (self.x_points[i+j] - self.x_points[i])
        
        self.tabela_dd = nova_tabela
        self.coeficientes = self.tabela_dd[0, :]
    
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Avalia o polinômio interpolador.
        """
        if self.coeficientes is None:
            raise ValueError("Nenhum ponto adicionado")
        
        return polinomio_newton(x, self.x_points, self.coeficientes)
    
    def get_polynomial(self) -> str:
        """Retorna representação simbólica do polinômio."""
        if self.coeficientes is None:
            return "Polinômio vazio"
        
        return newton_coeficientes_simbolicos(self.x_points, self.y_points)
    
    def clear(self):
        """Limpa todos os pontos."""
        self.x_points = []
        self.y_points = []
        self.coeficientes = None
        self.tabela_dd = None


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE ERROS
# ============================================================================

def erro_interpolacao_newton(x: float, 
                             x_i: List[float], 
                             coeficientes: np.ndarray,
                             f_derivada_max: float = None,
                             n_derivada: int = None) -> float:
    """
    Estima o erro da interpolação de Newton.
    
    Teorema: |f(x) - P(x)| ≤ |ω(x)| * max|f^(n+1)(ξ)| / (n+1)!
    """
    n = len(coeficientes) - 1
    
    # Calcula ω(x) = ∏(x - x_i)
    omega = 1.0
    for xi in x_i:
        omega *= (x - xi)
    
    if f_derivada_max is None:
        # Usa o próximo coeficiente como estimativa do erro
        if n + 1 < len(coeficientes):
            erro_estimado = abs(coeficientes[n+1] * omega)
        else:
            erro_estimado = abs(coeficientes[-1] * omega) / (n + 1)
    else:
        erro_estimado = abs(omega) * f_derivada_max / np.math.factorial(n + 1)
    
    return erro_estimado


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def plot_interpolacao_newton(x_i: List[float], 
                             y_i: List[float],
                             f_original: Callable = None,
                             x_range: Tuple[float, float] = None,
                             num_pontos: int = 500,
                             mostrar_erro: bool = True,
                             mostrar_diferencas: bool = False):
    """
    Plota a interpolação de Newton.
    """
    if x_range is None:
        x_range = (min(x_i) - 0.2*(max(x_i)-min(x_i)), 
                   max(x_i) + 0.2*(max(x_i)-min(x_i)))
    
    x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
    
    # Calcula coeficientes
    coef = diferencas_divididas(x_i, y_i)
    y_interp = polinomio_newton(x_plot, x_i, coef)
    
    if mostrar_diferencas:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_main = axes[0, 0]
        ax_error = axes[0, 1]
        ax_dd = axes[1, 0]
        ax_resid = axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2 if mostrar_erro and f_original else 1, 
                                figsize=(14, 5))
        if not mostrar_erro or f_original is None:
            axes = [axes[0]] if not isinstance(axes, np.ndarray) else [axes]
        ax_main = axes[0]
        if len(axes) > 1:
            ax_error = axes[1]
    
    # Gráfico principal
    ax_main.plot(x_plot, y_interp, 'b-', linewidth=2, label='Polinômio de Newton')
    ax_main.plot(x_i, y_i, 'ro', markersize=8, label='Pontos de interpolação')
    
    if f_original:
        y_original = f_original(x_plot)
        ax_main.plot(x_plot, y_original, 'g--', linewidth=1.5, 
                    label='Função original', alpha=0.7)
    
    ax_main.set_xlabel('x', fontsize=12)
    ax_main.set_ylabel('y', fontsize=12)
    ax_main.set_title('Interpolação de Newton', fontsize=14)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Gráfico de erro
    if mostrar_erro and f_original:
        erro = np.abs(y_interp - f_original(x_plot))
        ax_error.semilogy(x_plot, erro, 'r-', linewidth=2)
        ax_error.set_xlabel('x', fontsize=12)
        ax_error.set_ylabel('Erro Absoluto (log)', fontsize=12)
        ax_error.set_title('Erro da Interpolação', fontsize=14)
        ax_error.grid(True, alpha=0.3)
        
        # Marca os pontos de interpolação (erro zero)
        erro_nos = np.abs(y_i - f_original(x_i)) if f_original else np.zeros_like(x_i)
        ax_error.plot(x_i, erro_nos, 'bo', markersize=6, label='Nós')
        ax_error.legend()
    
    # Tabela de diferenças divididas
    if mostrar_diferencas:
        n = len(x_i)
        dd = np.zeros((n, n))
        dd[:, 0] = y_i
        
        for j in range(1, n):
            for i in range(n - j):
                dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x_i[i+j] - x_i[i])
        
        # Mostra tabela como texto
        ax_dd.axis('tight')
        ax_dd.axis('off')
        
        cabecalho = ['x', 'f(x)'] + [f'Δ^{i}' for i in range(1, n)]
        dados = []
        for i in range(n):
            linha = [f'{x_i[i]:.4f}', f'{dd[i,0]:.4f}']
            for j in range(1, n-i):
                linha.append(f'{dd[i,j]:.4f}')
            dados.append(linha)
        
        tabela = ax_dd.table(cellText=dados, colLabels=cabecalho[:len(dados[0])],
                            loc='center', cellLoc='center')
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(10)
        ax_dd.set_title('Tabela de Diferenças Divididas', fontsize=12)
        
        # Resíduos
        ax_resid.plot(x_i, y_i - polinomio_newton(x_i, x_i, coef), 
                     'go-', linewidth=2, markersize=6)
        ax_resid.axhline(y=0, color='r', linestyle='--')
        ax_resid.set_xlabel('x')
        ax_resid.set_ylabel('Resíduo')
        ax_resid.set_title('Resíduos nos Nós (devem ser zero)')
        ax_resid.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return coef


def plot_tabela_diferencas(x_i: List[float], y_i: List[float]):
    """
    Visualiza a tabela de diferenças divididas.
    """
    n = len(x_i)
    dd = np.zeros((n, n))
    dd[:, 0] = y_i
    
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i+1, j-1] - dd[i, j-1]) / (x_i[i+j] - x_i[i])
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepara dados para tabela
    col_labels = ['x', 'f(x)'] + [f'f[x{i},...,x{i+j}]' for j in range(1, n)]
    cell_text = []
    
    for i in range(n):
        row = [f'{x_i[i]:.6f}', f'{dd[i,0]:.6f}']
        for j in range(1, n - i):
            row.append(f'{dd[i,j]:.6f}')
        cell_text.append(row)
    
    # Cria tabela
    table = ax.table(cellText=cell_text, colLabels=col_labels[:len(cell_text[0])],
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Formatação
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif j == 0 or j == 1:
            cell.set_facecolor('#e8e8e8')
    
    ax.set_title('Tabela de Diferenças Divididas de Newton', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    return dd


# ============================================================================
# EXEMPLOS E APLICAÇÕES
# ============================================================================

def exemplo_basico():
    """Exemplo básico de interpolação de Newton."""
    print("=" * 70)
    print("EXEMPLO 1: Interpolação de Newton - Conceitos Básicos")
    print("=" * 70)
    
    # Dados
    x = [0, 1, 2, 3]
    y = [1, 2, 0, 2]
    
    print("Pontos de interpolação:")
    for xi, yi in zip(x, y):
        print(f"  ({xi}, {yi})")
    
    # Calcula diferenças divididas
    coef = diferencas_divididas(x, y)
    print(f"\nCoeficientes das diferenças divididas: {coef}")
    
    # Polinômio simbólico
    print(f"\nPolinômio: {newton_coeficientes_simbolicos(x, y)}")
    
    # Avalia em novos pontos
    x_teste = [0.5, 1.5, 2.5]
    for xt in x_teste:
        y_interp = polinomio_newton(xt, x, coef)
        print(f"P({xt}) = {y_interp:.4f}")
    
    # Visualização
    plot_interpolacao_newton(x, y)
    plot_tabela_diferencas(x, y)
    
    return coef


def exemplo_funcao_seno():
    """Exemplo com função seno."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Interpolação de Newton - Função Seno")
    print("=" * 70)
    
    # Pontos igualmente espaçados
    x_i = np.linspace(0, np.pi, 5)
    y_i = np.sin(x_i)
    
    print(f"Usando {len(x_i)} pontos igualmente espaçados")
    
    # Calcula polinômio
    coef = diferencas_divididas(x_i, y_i)
    print(f"\nCoeficientes: {coef}")
    
    # Testa em alguns pontos
    x_teste = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
    print("\nTeste de interpolação:")
    print(f"{'x':^10} {'sin(x)':^12} {'P(x)':^12} {'Erro':^12}")
    print("-" * 50)
    
    for xt in x_teste:
        y_exato = np.sin(xt)
        y_interp = polinomio_newton(xt, x_i, coef)
        erro = abs(y_interp - y_exato)
        print(f"{xt:^10.4f} {y_exato:^12.6f} {y_interp:^12.6f} {erro:^12.2e}")
    
    # Visualização
    plot_interpolacao_newton(x_i, y_i, np.sin, x_range=(0, np.pi))
    
    return coef


def exemplo_adicao_dinamica():
    """Exemplo de adição dinâmica de pontos."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Adição Dinâmica de Pontos (Fórmula de Newton)")
    print("=" * 70)
    
    # Cria interpolador
    interpolador = NewtonInterpolator()
    
    # Adiciona pontos gradualmente
    pontos = [(0, 1), (1, 2), (2, 0), (3, 2)]
    
    print("Adicionando pontos gradualmente:")
    for x, y in pontos:
        interpolador.add_point(x, y)
        print(f"\nApós adicionar ({x}, {y}):")
        print(f"  Polinômio: {interpolador.get_polynomial()}")
        
        # Testa em x = 1.5
        valor = interpolador(1.5)
        print(f"  P(1.5) = {valor:.4f}")
    
    # Visualização
    x_plot = np.linspace(-0.5, 3.5, 200)
    y_plot = interpolador(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio final')
    
    # Mostra pontos adicionados sequencialmente
    x_points = interpolador.x_points
    y_points = interpolador.y_points
    
    # Cores para diferentes estágios
    cores = plt.cm.viridis(np.linspace(0, 1, len(x_points)))
    
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        plt.plot(x, y, 'o', color=cores[i], markersize=10, 
                label=f'Ponto {i+1}: ({x}, {y})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolação de Newton - Adição Dinâmica de Pontos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return interpolador


def exemplo_diferencas_finitas():
    """Exemplo com diferenças finitas (Gregory-Newton)."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Fórmula de Gregory-Newton (Diferenças Finitas)")
    print("=" * 70)
    
    # Dados igualmente espaçados
    x = np.linspace(0, 2, 5)
    y = np.exp(x)
    
    print(f"Pontos igualmente espaçados: {x}")
    print(f"Valores: {y}")
    
    # Interpolação progressiva (início)
    x_interp = 0.5
    y_prog = newton_gregory_forward(x, y, x_interp)
    y_exato = np.exp(x_interp)
    
    print(f"\nInterpolação progressiva (x = {x_interp}):")
    print(f"  Valor interpolado: {y_prog:.8f}")
    print(f"  Valor exato: {y_exato:.8f}")
    print(f"  Erro: {abs(y_prog - y_exato):.2e}")
    
    # Interpolação regressiva (final)
    x_interp = 1.5
    y_reg = newton_gregory_backward(x, y, x_interp)
    y_exato = np.exp(x_interp)
    
    print(f"\nInterpolação regressiva (x = {x_interp}):")
    print(f"  Valor interpolado: {y_reg:.8f}")
    print(f"  Valor exato: {y_exato:.8f}")
    print(f"  Erro: {abs(y_reg - y_exato):.2e}")
    
    # Visualização comparativa
    x_plot = np.linspace(0, 2, 100)
    y_true = np.exp(x_plot)
    
    # Interpola em todos os pontos
    y_newton = []
    for xt in x_plot:
        if xt <= x[2]:  # Usa progressiva para primeira metade
            y_newton.append(newton_gregory_forward(x, y, xt))
        else:  # Usa regressiva para segunda metade
            y_newton.append(newton_gregory_backward(x, y, xt))
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_true, 'g-', linewidth=2, label='e^x (exato)')
    plt.plot(x_plot, y_newton, 'b--', linewidth=2, label='Gregory-Newton')
    plt.plot(x, y, 'ro', markersize=8, label='Pontos dados')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fórmula de Gregory-Newton para Pontos Igualmente Espaçados')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def exemplo_fenomeno_runge():
    """Demonstra o fenômeno de Runge com interpolação de Newton."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Fenômeno de Runge - Interpolação de Newton")
    print("=" * 70)
    
    def runge(x):
        return 1 / (1 + 25 * x**2)
    
    # Comparação com diferentes números de pontos
    n_pontos_list = [5, 10, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, n in enumerate(n_pontos_list):
        # Pontos igualmente espaçados
        x_i = np.linspace(-1, 1, n)
        y_i = runge(x_i)
        
        # Interpolação
        coef = diferencas_divididas(x_i, y_i)
        x_plot = np.linspace(-1, 1, 500)
        y_interp = polinomio_newton(x_plot, x_i, coef)
        y_true = runge(x_plot)
        
        # Plota
        ax = axes[idx]
        ax.plot(x_plot, y_true, 'g-', linewidth=2, label='Função original')
        ax.plot(x_plot, y_interp, 'r--', linewidth=1.5, label=f'Newton (n={n})')
        ax.plot(x_i, y_i, 'bo', markersize=5, label='Nós')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'{n} pontos igualmente espaçados')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calcula erro máximo
        erro_max = np.max(np.abs(y_interp - y_true))
        print(f"n = {n}: Erro máximo = {erro_max:.2e}")
    
    plt.tight_layout()
    plt.show()


def exemplo_comparacao_metodos():
    """Compara interpolação de Newton com Lagrange e Spline."""
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Comparação de Métodos de Interpolação")
    print("=" * 70)
    
    # Função de teste
    f = lambda x: np.sin(2 * np.pi * x) * np.exp(-x)
    
    # Pontos de interpolação
    n_pontos = 8
    x_i = np.linspace(0, 2, n_pontos)
    y_i = f(x_i)
    
    # Interpolação de Newton
    coef_newton = diferencas_divididas(x_i, y_i)
    
    # Interpolação de Lagrange (via numpy)
    from numpy.polynomial import lagrange
    p_lagrange = lagrange.Lagrange.fit(x_i, y_i, n_pontos-1)
    
    # Spline cúbica
    from scipy import interpolate
    spline = interpolate.CubicSpline(x_i, y_i)
    
    # Avaliação
    x_plot = np.linspace(0, 2, 500)
    y_true = f(x_plot)
    y_newton = polinomio_newton(x_plot, x_i, coef_newton)
    y_lagrange = p_lagrange(x_plot)
    y_spline = spline(x_plot)
    
    # Cálculo de erros
    erro_newton = np.abs(y_newton - y_true)
    erro_lagrange = np.abs(y_lagrange - y_true)
    erro_spline = np.abs(y_spline - y_true)
    
    print(f"Erro máximo (Newton): {np.max(erro_newton):.2e}")
    print(f"Erro máximo (Lagrange): {np.max(erro_lagrange):.2e}")
    print(f"Erro máximo (Spline): {np.max(erro_spline):.2e}")
    
    # Visualização
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico principal
    ax1 = axes[0, 0]
    ax1.plot(x_plot, y_true, 'k-', linewidth=2, label='Função original')
    ax1.plot(x_plot, y_newton, 'b--', linewidth=1.5, label='Newton')
    ax1.plot(x_plot, y_lagrange, 'g--', linewidth=1.5, label='Lagrange')
    ax1.plot(x_plot, y_spline, 'r--', linewidth=1.5, label='Spline cúbica')
    ax1.plot(x_i, y_i, 'ro', markersize=6, label='Nós')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Comparação de Métodos de Interpolação')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Erros
    ax2 = axes[0, 1]
    ax2.semilogy(x_plot, erro_newton, 'b-', label='Newton', linewidth=2)
    ax2.semilogy(x_plot, erro_lagrange, 'g-', label='Lagrange', linewidth=2)
    ax2.semilogy(x_plot, erro_spline, 'r-', label='Spline', linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Erro (log)')
    ax2.set_title('Erro de Interpolação')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Resíduos (diferença entre métodos)
    ax3 = axes[1, 0]
    ax3.plot(x_plot, y_newton - y_lagrange, 'b-', linewidth=2)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Diferença')
    ax3.set_title('Newton - Lagrange')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(x_plot, y_newton - y_spline, 'r-', linewidth=2)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Diferença')
    ax4.set_title('Newton - Spline')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def exemplo_aplicacao_engenharia():
    """Exemplo de aplicação em engenharia."""
    print("\n" + "=" * 70)
    print("EXEMPLO 7: Aplicação em Engenharia - Perfil de Temperatura")
    print("=" * 70)
    
    # Dados experimentais: temperatura em função do tempo
    tempo = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    temperatura = np.array([20, 25, 33, 40, 48, 55, 61, 66, 70, 73, 75])
    
    print("Dados experimentais:")
    print(f"Tempo (h): {tempo}")
    print(f"Temperatura (°C): {temperatura}")
    
    # Interpolação de Newton
    coef = diferencas_divididas(tempo, temperatura)
    
    # Previsões
    tempos_interp = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
    temperaturas_interp = polinomio_newton(tempos_interp, tempo, coef)
    
    print("\nPrevisões:")
    for t, temp in zip(tempos_interp, temperaturas_interp):
        print(f"  t = {t:.1f} h → T = {temp:.1f} °C")
    
    # Visualização
    x_plot = np.linspace(0, 10, 200)
    y_plot = polinomio_newton(x_plot, tempo, coef)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Polinômio de Newton')
    plt.plot(tempo, temperatura, 'ro', markersize=8, label='Dados experimentais')
    plt.plot(tempos_interp, temperaturas_interp, 'gs', markersize=10, 
             label='Pontos interpolados')
    plt.xlabel('Tempo (horas)', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.title('Interpolação de Newton - Perfil de Temperatura', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adiciona anotações
    for t, temp in zip(tempos_interp, temperaturas_interp):
        plt.annotate(f'({t:.1f}, {temp:.1f})', 
                    xy=(t, temp), xytext=(10, 10),
                    textcoords='offset points', fontsize=9)
    
    plt.show()
    
    # Verifica monotonicidade
    derivada = np.gradient(y_plot, x_plot)
    print(f"\nA temperatura está aumentando? {np.all(derivada > 0)}")
    
    return coef


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Executa exemplos
    exemplo_basico()
    exemplo_funcao_seno()
    exemplo_adicao_dinamica()
    exemplo_diferencas_finitas()
    exemplo_fenomeno_runge()
    exemplo_comparacao_metodos()
    exemplo_aplicacao_engenharia()
    
    print("\n" + "=" * 70)
    print("RESUMO DAS VANTAGENS DO MÉTODO DE NEWTON")
    print("=" * 70)
    print("""
    1. Fácil adição de novos pontos (fórmula incremental)
    2. Custo computacional O(n²) para coeficientes
    3. Avaliação eficiente O(n) com algoritmo de Horner
    4. Tabela de diferenças divididas fornece informação sobre o erro
    5. Natural para pontos não igualmente espaçados
    6. Fórmulas de Gregory-Newton para pontos igualmente espaçados
    """)