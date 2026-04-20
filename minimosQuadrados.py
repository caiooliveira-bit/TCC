import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union, Optional
from scipy import stats
import warnings

# ============================================================================
# IMPLEMENTAÇÃO BÁSICA DOS MÍNIMOS QUADRADOS
# ============================================================================

def ajuste_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, dict]:
    """
    Ajuste linear por mínimos quadrados: y = a + bx
    
    Parâmetros:
    x : array de valores independentes
    y : array de valores dependentes
    
    Retorna:
    a : coeficiente linear (intercepto)
    b : coeficiente angular (inclinação)
    stats : dicionário com estatísticas do ajuste
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    n = len(x)
    
    # Cálculo das somas
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    # Cálculo dos coeficientes
    denominador = n * sum_x2 - sum_x ** 2
    
    if abs(denominador) < 1e-10:
        raise ValueError("Dados insuficientes para ajuste linear")
    
    b = (n * sum_xy - sum_x * sum_y) / denominador
    a = (sum_y - b * sum_x) / n
    
    # Estatísticas do ajuste
    y_pred = a + b * x
    residuos = y - y_pred
    
    # Soma dos quadrados
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuos ** 2)
    ss_regression = ss_total - ss_residual
    
    # Coeficiente de determinação R²
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Erro padrão
    sigma2 = ss_residual / (n - 2) if n > 2 else 0
    erro_padrao_b = np.sqrt(sigma2 / (sum_x2 - sum_x**2 / n)) if n > 2 else 0
    erro_padrao_a = np.sqrt(sigma2 * (1/n + sum_x**2/(n * (sum_x2 - sum_x**2/n)))) if n > 2 else 0
    
    stats_dict = {
        'r_squared': r_squared,
        'r': np.sqrt(r_squared) * (1 if b > 0 else -1),
        'ss_total': ss_total,
        'ss_residual': ss_residual,
        'ss_regression': ss_regression,
        'erro_padrao_a': erro_padrao_a,
        'erro_padrao_b': erro_padrao_b,
        'sigma2': sigma2,
        'residuos': residuos,
        'y_pred': y_pred,
        'n': n
    }
    
    return a, b, stats_dict


def ajuste_polinomial(x: np.ndarray, y: np.ndarray, grau: int) -> np.poly1d:
    """
    Ajuste polinomial por mínimos quadrados.
    
    Parâmetros:
    x : array de valores independentes
    y : array de valores dependentes
    grau : grau do polinômio
    
    Retorna:
    p : polinômio ajustado (objeto poly1d)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Constrói matriz de Vandermonde
    V = np.vander(x, grau + 1, increasing=True)
    
    # Resolve sistema normal (V^T V) c = V^T y
    # Usando decomposição QR para maior estabilidade
    Q, R = np.linalg.qr(V, mode='reduced')
    c = np.linalg.solve(R, Q.T @ y)
    
    # Inverte a ordem dos coeficientes (poly1d espera do maior grau para o menor)
    coefs = c[::-1]
    
    return np.poly1d(coefs)


def ajuste_polinomial_normal(x: np.ndarray, y: np.ndarray, grau: int) -> np.poly1d:
    """
    Ajuste polinomial usando equações normais (alternativa mais simples).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Constrói sistema normal
    A = np.zeros((grau + 1, grau + 1))
    b = np.zeros(grau + 1)
    
    for i in range(grau + 1):
        for j in range(grau + 1):
            A[i, j] = np.sum(x ** (i + j))
        b[i] = np.sum(y * (x ** i))
    
    # Resolve sistema
    c = np.linalg.solve(A, b)
    
    # poly1d espera coeficientes do maior grau para o menor
    return np.poly1d(c[::-1])


def ajuste_multilinear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ajuste linear multivariado: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    
    Parâmetros:
    X : matriz de variáveis independentes (n_samples, n_features)
    y : vetor de valores dependentes (n_samples,)
    
    Retorna:
    beta : coeficientes [β₀, β₁, ..., βₙ]
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Adiciona coluna de 1s para o intercepto
    X_design = np.column_stack([np.ones(len(X)), X])
    
    # Resolve por mínimos quadrados: (X^T X) β = X^T y
    # Usando decomposição QR para estabilidade numérica
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    
    return beta


def ajuste_ponderado(x: np.ndarray, y: np.ndarray, pesos: np.ndarray, grau: int = 1) -> np.poly1d:
    """
    Ajuste polinomial ponderado por mínimos quadrados.
    
    Parâmetros:
    x : valores independentes
    y : valores dependentes
    pesos : pesos para cada ponto
    grau : grau do polinômio
    
    Retorna:
    p : polinômio ajustado
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    pesos = np.array(pesos, dtype=float)
    
    # Constrói sistema ponderado
    V = np.vander(x, grau + 1, increasing=True)
    
    # Aplica pesos
    W = np.diag(np.sqrt(pesos))
    Vw = W @ V
    yw = W @ y
    
    # Resolve sistema
    coefs = np.linalg.lstsq(Vw, yw, rcond=None)[0]
    
    return np.poly1d(coefs[::-1])


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE QUALIDADE DO AJUSTE
# ============================================================================

def analise_ajuste(x: np.ndarray, y: np.ndarray, modelo: Callable, params: np.ndarray = None) -> dict:
    """
    Análise completa da qualidade do ajuste.
    
    Parâmetros:
    x : valores independentes
    y : valores dependentes
    modelo : função do modelo (ex: lambda x, a, b: a + b*x)
    params : parâmetros do modelo (se None, calcula por mínimos quadrados)
    
    Retorna:
    dict : métricas de qualidade do ajuste
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if params is None:
        # Para modelo linear simples, usa ajuste linear
        if modelo.__code__.co_argcount == 3:  # modelo com 2 parâmetros
            a, b, _ = ajuste_linear(x, y)
            params = [a, b]
            y_pred = modelo(x, a, b)
        else:
            raise ValueError("Forneça os parâmetros ou use um modelo linear simples")
    else:
        y_pred = modelo(x, *params)
    
    # Cálculo das métricas
    residuos = y - y_pred
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuos ** 2)
    ss_regression = ss_total - ss_residual
    
    # R² e R² ajustado
    n = len(y)
    p = len(params)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared
    
    # Estatísticas dos resíduos
    residuos_mean = np.mean(residuos)
    residuos_std = np.std(residuos)
    residuos_skew = stats.skew(residuos) if len(residuos) > 2 else 0
    residuos_kurtosis = stats.kurtosis(residuos) if len(residuos) > 3 else 0
    
    # Teste de normalidade dos resíduos (Shapiro-Wilk)
    if 3 <= n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuos)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Erro padrão da estimativa
    se_estimate = np.sqrt(ss_residual / (n - p - 1)) if n > p + 1 else np.nan
    
    # AIC e BIC (critérios de informação)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_residual / n) + 1)
    aic = 2 * p - 2 * log_likelihood
    bic = p * np.log(n) - 2 * log_likelihood
    
    resultados = {
        'params': params,
        'r_squared': r_squared,
        'r_squared_ajustado': r_squared_adj,
        'ss_total': ss_total,
        'ss_residual': ss_residual,
        'ss_regression': ss_regression,
        'residuos': residuos,
        'y_pred': y_pred,
        'residuos_mean': residuos_mean,
        'residuos_std': residuos_std,
        'residuos_skew': residuos_skew,
        'residuos_kurtosis': residuos_kurtosis,
        'erro_padrao_estimativa': se_estimate,
        'shapiro_stat': shapiro_stat,
        'shapiro_p_value': shapiro_p,
        'aic': aic,
        'bic': bic,
        'n': n,
        'p': p
    }
    
    return resultados


def comparar_modelos(x: np.ndarray, y: np.ndarray, modelos: List[Tuple[Callable, int]]) -> pd.DataFrame:
    """
    Compara diferentes modelos de regressão.
    
    Parâmetros:
    x : valores independentes
    y : valores dependentes
    modelos : lista de tuplas (função_modelo, grau) ou (função_modelo, params)
    
    Retorna:
    DataFrame com comparação dos modelos
    """
    try:
        import pandas as pd
    except ImportError:
        print("Instale pandas para usar esta função: pip install pandas")
        return None
    
    resultados = []
    
    for i, (modelo, param) in enumerate(modelos):
        if isinstance(param, int):  # Grau polinomial
            p = ajuste_polinomial(x, y, param)
            y_pred = p(x)
            params = p.coef
            nome = f"Polinomial (grau {param})"
        else:  # Parâmetros fornecidos
            y_pred = modelo(x, *param)
            params = param
            nome = f"Modelo {i+1}"
        
        # Calcula métricas
        residuos = y - y_pred
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(residuos ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        n = len(y)
        p_count = len(params)
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p_count - 1) if n > p_count + 1 else r_squared
        
        aic = n * np.log(ss_residual / n) + 2 * p_count
        bic = n * np.log(ss_residual / n) + p_count * np.log(n)
        
        resultados.append({
            'Modelo': nome,
            'R²': r_squared,
            'R²_ajustado': r_squared_adj,
            'RMSE': np.sqrt(np.mean(residuos**2)),
            'MAE': np.mean(np.abs(residuos)),
            'AIC': aic,
            'BIC': bic,
            'Params': params
        })
    
    df = pd.DataFrame(resultados)
    df = df.sort_values('R²_ajustado', ascending=False)
    
    return df


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO
# ============================================================================

def plot_ajuste(x: np.ndarray, y: np.ndarray, modelo: Union[Callable, np.poly1d], 
                x_range: Tuple[float, float] = None, num_pontos: int = 200,
                titulo: str = "Ajuste por Mínimos Quadrados"):
    """
    Plota os dados e o modelo ajustado.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if x_range is None:
        x_range = (np.min(x) - 0.1 * (np.max(x) - np.min(x)),
                   np.max(x) + 0.1 * (np.max(x) - np.min(x)))
    
    x_plot = np.linspace(x_range[0], x_range[1], num_pontos)
    
    if isinstance(modelo, np.poly1d):
        y_plot = modelo(x_plot)
        y_pred = modelo(x)
    else:
        y_plot = modelo(x_plot)
        y_pred = modelo(x)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico principal
    ax1 = axes[0]
    ax1.scatter(x, y, alpha=0.6, s=50, label='Dados', color='blue')
    ax1.plot(x_plot, y_plot, 'r-', linewidth=2, label='Ajuste')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(titulo, fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de resíduos
    ax2 = axes[1]
    residuos = y - y_pred
    ax2.scatter(x, residuos, alpha=0.6, s=50, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Resíduos', fontsize=12)
    ax2.set_title('Análise de Resíduos', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Adiciona banda de confiança (opcional)
    # std_res = np.std(residuos)
    # ax2.fill_between(x, -2*std_res, 2*std_res, alpha=0.2, color='gray')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


def plot_diagnostico(resultados: dict):
    """
    Gráficos de diagnóstico para análise de regressão.
    """
    residuos = resultados['residuos']
    y_pred = resultados['y_pred']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Resíduos vs Valores Ajustados
    axes[0, 0].scatter(y_pred, residuos, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Valores Ajustados')
    axes[0, 0].set_ylabel('Resíduos')
    axes[0, 0].set_title('Resíduos vs Ajustados')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot dos resíduos
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot dos Resíduos')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograma dos resíduos
    axes[1, 0].hist(residuos, bins='auto', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Resíduos')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].set_title('Histograma dos Resíduos')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Resíduos vs Ordem
    axes[1, 1].plot(residuos, 'o-', alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observação')
    axes[1, 1].set_ylabel('Resíduos')
    axes[1, 1].set_title('Resíduos em Ordem')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_comparacao_modelos(x: np.ndarray, y: np.ndarray, modelos: List[Union[np.poly1d, Callable]], 
                            labels: List[str] = None):
    """
    Compara visualmente diferentes modelos.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    x_plot = np.linspace(np.min(x) - 0.1, np.max(x) + 0.1, 500)
    
    plt.figure(figsize=(12, 8))
    
    # Plota dados
    plt.scatter(x, y, alpha=0.6, s=50, color='black', label='Dados', zorder=5)
    
    # Plota cada modelo
    cores = plt.cm.tab10(np.linspace(0, 1, len(modelos)))
    
    for i, modelo in enumerate(modelos):
        label = labels[i] if labels else f'Modelo {i+1}'
        if isinstance(modelo, np.poly1d):
            y_plot = modelo(x_plot)
        else:
            y_plot = modelo(x_plot)
        
        # Calcula R²
        if isinstance(modelo, np.poly1d):
            y_pred = modelo(x)
        else:
            y_pred = modelo(x)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        plt.plot(x_plot, y_plot, color=cores[i], linewidth=2, 
                label=f'{label} (R² = {r2:.4f})')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Comparação de Modelos de Regressão', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXEMPLOS E APLICAÇÕES
# ============================================================================

def exemplo_ajuste_linear():
    """Exemplo de ajuste linear simples."""
    print("=" * 70)
    print("EXEMPLO 1: Ajuste Linear Simples")
    print("=" * 70)
    
    # Dados com relação linear
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = 2.5 + 1.8 * x
    y = y_true + np.random.normal(0, 1, len(x))  # Adiciona ruído
    
    # Ajuste linear
    a, b, stats = ajuste_linear(x, y)
    
    print(f"Modelo ajustado: y = {a:.4f} + {b:.4f}x")
    print(f"R² = {stats['r_squared']:.4f}")
    print(f"Erro padrão de a: {stats['erro_padrao_a']:.4f}")
    print(f"Erro padrão de b: {stats['erro_padrao_b']:.4f}")
    
    # Análise completa
    def modelo_linear(x, a, b):
        return a + b * x
    
    analise = analise_ajuste(x, y, modelo_linear, [a, b])
    print(f"\nAnálise dos resíduos:")
    print(f"  Média: {analise['residuos_mean']:.4f}")
    print(f"  Desvio padrão: {analise['residuos_std']:.4f}")
    print(f"  Assimetria: {analise['residuos_skew']:.4f}")
    print(f"  Curtose: {analise['residuos_kurtosis']:.4f}")
    
    # Visualização
    plot_ajuste(x, y, lambda x: a + b*x, titulo="Ajuste Linear")
    plot_diagnostico(analise)
    
    return a, b, stats


def exemplo_ajuste_polinomial():
    """Exemplo de ajuste polinomial."""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: Ajuste Polinomial")
    print("=" * 70)
    
    # Dados com relação quadrática
    np.random.seed(42)
    x = np.linspace(-3, 3, 30)
    y_true = 1 - 2*x + 0.5*x**2
    y = y_true + np.random.normal(0, 0.5, len(x))
    
    # Ajuste para diferentes graus
    graus = [1, 2, 3]
    modelos = []
    
    for grau in graus:
        p = ajuste_polinomial(x, y, grau)
        modelos.append(p)
        print(f"\nGrau {grau}:")
        print(f"  Polinômio: {p}")
        
        # Calcula R²
        y_pred = p(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"  R² = {r2:.4f}")
    
    # Compara modelos
    plot_comparacao_modelos(x, y, modelos, [f'Grau {g}' for g in graus])
    
    # Melhor modelo (grau 2, que gerou os dados)
    p_quad = ajuste_polinomial(x, y, 2)
    plot_ajuste(x, y, p_quad, titulo="Ajuste Quadrático")
    
    return modelos


def exemplo_regressao_multilinear():
    """Exemplo de regressão linear múltipla."""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: Regressão Linear Múltipla")
    print("=" * 70)
    
    # Dados sintéticos: y = 2 + 3x₁ - 1.5x₂ + erro
    np.random.seed(42)
    n_samples = 100
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y_true = 2 + 3*X1 - 1.5*X2
    y = y_true + np.random.normal(0, 0.3, n_samples)
    
    X = np.column_stack([X1, X2])
    
    # Ajuste multilinear
    beta = ajuste_multilinear(X, y)
    
    print(f"Modelo ajustado: y = {beta[0]:.4f} + {beta[1]:.4f}x₁ + {beta[2]:.4f}x₂")
    print(f"Coeficientes verdadeiros: β₀=2, β₁=3, β₂=-1.5")
    
    # Previsões
    y_pred = beta[0] + beta[1]*X1 + beta[2]*X2
    
    # Estatísticas
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"R² = {r2:.4f}")
    
    # Visualização (para 2 variáveis, podemos plotar em 3D)
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # Gráfico 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X1, X2, y, c='blue', alpha=0.6)
    
    # Superfície do modelo
    x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    y_grid = beta[0] + beta[1]*x1_grid + beta[2]*x2_grid
    ax1.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color='red')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('y')
    ax1.set_title('Regressão Linear Múltipla')
    
    # Gráfico de resíduos
    ax2 = fig.add_subplot(122)
    residuos = y - y_pred
    ax2.scatter(y_pred, residuos, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Valores Preditos')
    ax2.set_ylabel('Resíduos')
    ax2.set_title('Resíduos vs Preditos')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return beta


def exemplo_ajuste_ponderado():
    """Exemplo de ajuste ponderado."""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: Ajuste Ponderado")
    print("=" * 70)
    
    # Dados com diferentes níveis de confiança
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = 2*x + 1 + np.random.normal(0, x, len(x))  # Variância proporcional a x
    
    # Pesos inversamente proporcionais à variância
    pesos = 1 / x
    
    # Ajuste sem pesos
    p_sem_peso = ajuste_polinomial(x, y, 1)
    
    # Ajuste com pesos
    p_com_peso = ajuste_ponderado(x, y, pesos, 1)
    
    print("Comparação dos ajustes:")
    print(f"Sem pesos:    y = {p_sem_peso[1]:.4f}x + {p_sem_peso[0]:.4f}")
    print(f"Com pesos:    y = {p_com_peso[1]:.4f}x + {p_com_peso[0]:.4f}")
    print(f"Verdadeiro:   y = 2.0000x + 1.0000")
    
    # Visualização
    x_plot = np.linspace(0, 11, 200)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, s=50*pesos/np.max(pesos), label='Dados (tamanho = peso)')
    plt.plot(x_plot, p_sem_peso(x_plot), 'b-', label='Sem pesos', linewidth=2)
    plt.plot(x_plot, p_com_peso(x_plot), 'r-', label='Com pesos', linewidth=2)
    plt.plot(x_plot, 2*x_plot + 1, 'g--', label='Verdadeiro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ajuste Linear Ponderado vs Não Ponderado')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return p_sem_peso, p_com_peso


def exemplo_aplicacao_real():
    """Exemplo com dados reais (propriedade de materiais)."""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: Aplicação Real - Lei de Hooke")
    print("=" * 70)
    
    # Dados experimentais: Força (N) vs Deformação (mm)
    forca = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    deformacao = np.array([0, 0.21, 0.39, 0.62, 0.79, 1.01, 
                           1.21, 1.39, 1.62, 1.79, 2.01])
    
    print("Dados experimentais (Lei de Hooke):")
    print("Força (N) | Deformação (mm)")
    for f, d in zip(forca, deformacao):
        print(f"  {f:3.0f}     |    {d:.3f}")
    
    # Ajuste linear (F = k * x)
    a, b, stats = ajuste_linear(forca, deformacao)
    
    # k é o coeficiente angular (b)
    k = b
    print(f"\nConstante elástica (k) = {k:.3f} N/mm")
    print(f"R² = {stats['r_squared']:.4f}")
    
    # Previsões
    forca_teste = np.array([25, 55, 85])
    deformacao_pred = a + b * forca_teste
    
    print("\nPrevisões:")
    for f, d in zip(forca_teste, deformacao_pred):
        print(f"Para F = {f:.0f} N, deformação prevista = {d:.3f} mm")
    
    # Visualização
    plot_ajuste(forca, deformacao, lambda x: a + b*x, 
                titulo="Lei de Hooke - Determinação da Constante Elástica")
    
    return k, stats


# ============================================================================
# CLASSE PARA REGRESSÃO (similar ao scikit-learn)
# ============================================================================

class MinimosQuadrados:
    """
    Classe para regressão por mínimos quadrados com interface similar ao sklearn.
    """
    
    def __init__(self, grau: int = 1, fit_intercept: bool = True):
        """
        Parâmetros:
        grau : grau do polinômio (para regressão polinomial)
        fit_intercept : se deve incluir intercepto
        """
        self.grau = grau
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta o modelo aos dados.
        
        Parâmetros:
        X : matriz de features (n_samples, n_features)
        y : vetor alvo (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        # Para regressão polinomial
        if self.grau > 1 and X.ndim == 1:
            # Cria features polinomiais
            X_poly = np.column_stack([X ** i for i in range(1, self.grau + 1)])
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X)), X_poly])
            else:
                X_design = X_poly
        else:
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X)), X])
            else:
                X_design = X
        
        # Resolve por mínimos quadrados
        self.coef_, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões para novos dados.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado. Chame fit primeiro.")
        
        X = np.array(X, dtype=float)
        
        if self.grau > 1 and X.ndim == 1:
            X_poly = np.column_stack([X ** i for i in range(1, self.grau + 1)])
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X)), X_poly])
            else:
                X_design = X_poly
        else:
            if self.fit_intercept:
                X_design = np.column_stack([np.ones(len(X)), X])
            else:
                X_design = X
        
        if self.fit_intercept:
            coef_full = np.concatenate([[self.intercept_], self.coef_])
        else:
            coef_full = self.coef_
        
        return X_design @ coef_full
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Retorna o R² do modelo.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def get_params(self) -> dict:
        """Retorna os parâmetros do modelo."""
        return {
            'coef_': self.coef_,
            'intercept_': self.intercept_,
            'grau': self.grau,
            'fit_intercept': self.fit_intercept
        }


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Executa todos os exemplos
    exemplo_ajuste_linear()
    exemplo_ajuste_polinomial()
    exemplo_regressao_multilinear()
    exemplo_ajuste_ponderado()
    exemplo_aplicacao_real()
    
    # Demonstração da classe MinimosQuadrados
    print("\n" + "=" * 70)
    print("EXEMPLO 6: Usando a Classe MinimosQuadrados")
    print("=" * 70)
    
    # Dados
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = 3*X + 2 + np.random.normal(0, 2, 50)
    
    # Cria e ajusta modelo
    model = MinimosQuadrados(grau=1, fit_intercept=True)
    model.fit(X, y)
    
    print(f"Intercepto: {model.intercept_:.4f}")
    print(f"Coefficiente: {model.coef_[0]:.4f}")
    print(f"R²: {model.score(X, y):.4f}")
    
    # Previsão
    X_new = np.array([5, 7, 9])
    y_pred = model.predict(X_new)
    print(f"\nPrevisões para X = {X_new}: {y_pred}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Dados')
    plt.plot(X, model.predict(X), 'r-', linewidth=2, label='Ajuste')
    plt.scatter(X_new, y_pred, color='green', s=100, marker='s', label='Previsões')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Regressão com Classe MinimosQuadrados')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()