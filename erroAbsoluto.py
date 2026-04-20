import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Callable

# ============================================================================
# FUNÇÕES BÁSICAS PARA CÁLCULO DE ERROS
# ============================================================================

def erro_absoluto(valor_aproximado: Union[float, np.ndarray], 
                  valor_exato: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula o erro absoluto: |valor_aproximado - valor_exato|
    
    Parâmetros:
    valor_aproximado : valor aproximado (escalar ou array)
    valor_exato : valor exato (escalar ou array)
    
    Retorna:
    erro_absoluto : erro absoluto (mesmo tipo da entrada)
    """
    return np.abs(valor_aproximado - valor_exato)


def erro_relativo(valor_aproximado: Union[float, np.ndarray], 
                  valor_exato: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula o erro relativo: |valor_aproximado - valor_exato| / |valor_exato|
    
    Parâmetros:
    valor_aproximado : valor aproximado (escalar ou array)
    valor_exato : valor exato (escalar ou array)
    
    Retorna:
    erro_relativo : erro relativo (mesmo tipo da entrada)
    
    Atenção:
    Se valor_exato for zero, retorna erro absoluto (evita divisão por zero)
    """
    valor_exato = np.array(valor_exato, dtype=float)
    valor_aproximado = np.array(valor_aproximado, dtype=float)
    
    # Evita divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        erro_rel = erro_absoluto(valor_aproximado, valor_exato) / np.abs(valor_exato)
        # Trata casos onde valor_exato é zero
        erro_rel = np.where(np.abs(valor_exato) < 1e-15, 
                           erro_absoluto(valor_aproximado, valor_exato), 
                           erro_rel)
    
    return erro_rel


def erro_percentual(valor_aproximado: Union[float, np.ndarray], 
                    valor_exato: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula o erro percentual: erro_relativo * 100%
    
    Parâmetros:
    valor_aproximado : valor aproximado
    valor_exato : valor exato
    
    Retorna:
    erro_percentual : erro em percentual
    """
    return erro_relativo(valor_aproximado, valor_exato) * 100


# ============================================================================
# FUNÇÕES PARA ANÁLISE DE ERROS EM MÉTODOS NUMÉRICOS
# ============================================================================

def analise_erros_solucao(A: np.ndarray, 
                          b: np.ndarray, 
                          x_exato: np.ndarray,
                          metodo_resolucao: Callable) -> dict:
    """
    Analisa erros absolutos e relativos na solução de sistemas lineares.
    
    Parâmetros:
    A : matriz do sistema
    b : vetor independente
    x_exato : solução exata
    metodo_resolucao : função que resolve o sistema (ex: lambda A,b: np.linalg.solve(A,b))
    
    Retorna:
    dict : dicionário com métricas de erro
    """
    # Calcula solução aproximada
    x_aprox = metodo_resolucao(A, b)
    
    # Calcula erros
    erro_abs = erro_absoluto(x_aprox, x_exato)
    erro_rel = erro_relativo(x_aprox, x_exato)
    
    # Métricas
    resultados = {
        'solucao_exata': x_exato,
        'solucao_aproximada': x_aprox,
        'erro_absoluto_medio': np.mean(erro_abs),
        'erro_absoluto_maximo': np.max(erro_abs),
        'erro_absoluto_norma2': np.linalg.norm(erro_abs),
        'erro_relativo_medio': np.mean(erro_rel),
        'erro_relativo_maximo': np.max(erro_rel),
        'erro_relativo_norma2': np.linalg.norm(erro_rel),
        'erro_percentual_medio': np.mean(erro_rel) * 100,
        'erro_percentual_maximo': np.max(erro_rel) * 100,
        'erros_absolutos_por_componente': erro_abs,
        'erros_relativos_por_componente': erro_rel
    }
    
    return resultados


def analise_convergencia_bissecao(f: Callable, 
                                  a: float, 
                                  b: float, 
                                  raiz_exata: float,
                                  max_iter: int = 50) -> dict:
    """
    Analisa a convergência do método da bisseção calculando erros.
    
    Parâmetros:
    f : função
    a, b : intervalo inicial
    raiz_exata : valor exato da raiz
    max_iter : número máximo de iterações
    
    Retorna:
    dict : histórico de erros
    """
    history = {
        'iteracoes': [],
        'aproximacoes': [],
        'erros_absolutos': [],
        'erros_relativos': [],
        'erros_percentuais': []
    }
    
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("Intervalo não contém raiz")
    
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        
        # Calcula erros
        erro_abs = erro_absoluto(c, raiz_exata)
        erro_rel = erro_relativo(c, raiz_exata)
        
        # Armazena histórico
        history['iteracoes'].append(i)
        history['aproximacoes'].append(c)
        history['erros_absolutos'].append(erro_abs)
        history['erros_relativos'].append(erro_rel)
        history['erros_percentuais'].append(erro_rel * 100)
        
        # Critério de parada
        if abs(fc) < 1e-12 or (b - a) / 2 < 1e-12:
            break
        
        # Atualiza intervalo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return history


def tabela_erros(aprox: List[float], 
                 exato: List[float], 
                 labels: List[str] = None) -> None:
    """
    Exibe uma tabela comparativa de erros absolutos e relativos.
    
    Parâmetros:
    aprox : valores aproximados
    exato : valores exatos
    labels : rótulos para cada valor
    """
    aprox = np.array(aprox)
    exato = np.array(exato)
    
    if labels is None:
        labels = [f"Valor {i+1}" for i in range(len(aprox))]
    
    print("=" * 80)
    print(f"{'Item':<15} {'Valor Exato':<15} {'Valor Aprox':<15} {'Erro Abs':<15} {'Erro Rel (%)':<15}")
    print("-" * 80)
    
    for i, label in enumerate(labels):
        ea = erro_absoluto(aprox[i], exato[i])
        er = erro_relativo(aprox[i], exato[i]) * 100
        print(f"{label:<15} {exato[i]:<15.8f} {aprox[i]:<15.8f} {ea:<15.2e} {er:<15.6f}")
    
    print("=" * 80)


# ============================================================================
# FUNÇÕES PARA VISUALIZAÇÃO DE ERROS
# ============================================================================

def plot_erros_convergencia(history: dict, 
                            titulo: str = "Análise de Erros") -> None:
    """
    Plota gráficos de convergência de erros.
    
    Parâmetros:
    history : histórico de erros (retornado por analise_convergencia_bissecao)
    titulo : título do gráfico
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(titulo, fontsize=14)
    
    iteracoes = history['iteracoes']
    
    # Gráfico 1: Erro absoluto
    axes[0, 0].semilogy(iteracoes, history['erros_absolutos'], 'b-o', linewidth=2)
    axes[0, 0].set_xlabel('Iterações')
    axes[0, 0].set_ylabel('Erro Absoluto (log)')
    axes[0, 0].set_title('Convergência do Erro Absoluto')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Erro relativo
    axes[0, 1].semilogy(iteracoes, history['erros_relativos'], 'r-o', linewidth=2)
    axes[0, 1].set_xlabel('Iterações')
    axes[0, 1].set_ylabel('Erro Relativo (log)')
    axes[0, 1].set_title('Convergência do Erro Relativo')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Aproximações vs Valor Exato
    axes[1, 0].plot(iteracoes, history['aproximacoes'], 'g-o', linewidth=2, label='Aproximação')
    axes[1, 0].axhline(y=history['aproximacoes'][-1], color='r', linestyle='--', 
                       label=f'Valor final: {history["aproximacoes"][-1]:.8f}')
    axes[1, 0].set_xlabel('Iterações')
    axes[1, 0].set_ylabel('Valor')
    axes[1, 0].set_title('Convergência da Aproximação')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Erro percentual
    axes[1, 1].semilogy(iteracoes, history['erros_percentuais'], 'm-o', linewidth=2)
    axes[1, 1].set_xlabel('Iterações')
    axes[1, 1].set_ylabel('Erro Percentual (%) (log)')
    axes[1, 1].set_title('Convergência do Erro Percentual')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_comparacao_erros(valores_exatos: List[float],
                          valores_aprox: List[float],
                          labels: List[str] = None) -> None:
    """
    Gráfico de barras comparando erros absolutos e relativos.
    """
    valores_exatos = np.array(valores_exatos)
    valores_aprox = np.array(valores_aprox)
    
    if labels is None:
        labels = [f"Item {i+1}" for i in range(len(valores_exatos))]
    
    erros_abs = erro_absoluto(valores_aprox, valores_exatos)
    erros_rel = erro_relativo(valores_aprox, valores_exatos) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de erros absolutos
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, valores_exatos, width, label='Valor Exato', alpha=0.7)
    ax1.bar(x + width/2, valores_aprox, width, label='Valor Aprox', alpha=0.7)
    ax1.set_xlabel('Itens')
    ax1.set_ylabel('Valores')
    ax1.set_title('Comparação de Valores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de erros
    ax2.bar(x - width/2, erros_abs, width, label='Erro Absoluto', alpha=0.7)
    ax2.bar(x + width/2, erros_rel, width, label='Erro Relativo (%)', alpha=0.7)
    ax2.set_xlabel('Itens')
    ax2.set_ylabel('Erros')
    ax2.set_title('Análise de Erros')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# FUNÇÕES PARA PROPAGAÇÃO DE ERROS
# ============================================================================

def propagacao_erro_soma(a: float, b: float, 
                         erro_a: float, erro_b: float) -> Tuple[float, float]:
    """
    Calcula erro absoluto e relativo para soma: c = a + b
    """
    c = a + b
    erro_abs_c = erro_absoluto(erro_a, 0) + erro_absoluto(erro_b, 0)  # Soma dos erros absolutos
    erro_rel_c = erro_relativo(erro_abs_c, c) if c != 0 else float('inf')
    return erro_abs_c, erro_rel_c


def propagacao_erro_produto(a: float, b: float,
                           erro_a: float, erro_b: float) -> Tuple[float, float]:
    """
    Calcula erro absoluto e relativo para produto: c = a * b
    """
    c = a * b
    erro_rel_c = erro_relativo(erro_a, a) + erro_relativo(erro_b, b)
    erro_abs_c = erro_rel_c * abs(c)
    return erro_abs_c, erro_rel_c


def propagacao_erro_divisao(a: float, b: float,
                           erro_a: float, erro_b: float) -> Tuple[float, float]:
    """
    Calcula erro absoluto e relativo para divisão: c = a / b
    """
    if b == 0:
        raise ValueError("Divisão por zero")
    
    c = a / b
    erro_rel_c = erro_relativo(erro_a, a) + erro_relativo(erro_b, b)
    erro_abs_c = erro_rel_c * abs(c)
    return erro_abs_c, erro_rel_c


# ============================================================================
# EXEMPLOS E TESTES
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("EXEMPLO 1: Erros Básicos")
    print("=" * 80)
    
    # Exemplos simples
    valores_exatos = [np.pi, np.e, np.sqrt(2), 1.0]
    valores_aprox = [3.1416, 2.7183, 1.4142, 0.9999]
    
    for exato, aprox in zip(valores_exatos, valores_aprox):
        ea = erro_absoluto(aprox, exato)
        er = erro_relativo(aprox, exato)
        ep = erro_percentual(aprox, exato)
        
        print(f"Exato: {exato:.8f}, Aprox: {aprox:.8f}")
        print(f"  Erro Absoluto: {ea:.2e}")
        print(f"  Erro Relativo: {er:.2e} ({ep:.4f}%)")
        print()
    
    print("\n" + "=" * 80)
    print("EXEMPLO 2: Tabela Comparativa de Erros")
    print("=" * 80)
    
    tabela_erros(valores_aprox, valores_exatos, 
                labels=['π', 'e', '√2', '1.0'])
    
    print("\n" + "=" * 80)
    print("EXEMPLO 3: Análise de Erros em Sistemas Lineares")
    print("=" * 80)
    
    # Sistema bem condicionado
    A_bom = np.array([[4, 1], [1, 3]])
    x_exato = np.array([1, 2])
    b = A_bom @ x_exato
    
    resultados = analise_erros_solucao(A_bom, b, x_exato, 
                                      lambda A, b: np.linalg.solve(A, b))
    
    print("Sistema bem condicionado:")
    print(f"  Erro absoluto médio: {resultados['erro_absoluto_medio']:.2e}")
    print(f"  Erro relativo médio: {resultados['erro_relativo_medio']:.2e}")
    print(f"  Erro percentual máximo: {resultados['erro_percentual_maximo']:.4f}%")
    
    # Sistema mal condicionado (matriz de Hilbert)
    n = 5
    A_mal = np.array([[1/(i+j+1) for j in range(n)] for i in range(n)])
    x_exato = np.ones(n)
    b_mal = A_mal @ x_exato
    
    resultados_mal = analise_erros_solucao(A_mal, b_mal, x_exato,
                                          lambda A, b: np.linalg.solve(A, b))
    
    print("\nSistema mal condicionado (Hilbert 5x5):")
    print(f"  Erro absoluto médio: {resultados_mal['erro_absoluto_medio']:.2e}")
    print(f"  Erro relativo médio: {resultados_mal['erro_relativo_medio']:.2e}")
    print(f"  Erro percentual máximo: {resultados_mal['erro_percentual_maximo']:.4f}%")
    
    print("\n" + "=" * 80)
    print("EXEMPLO 4: Convergência do Método da Bisseção")
    print("=" * 80)
    
    # f(x) = x² - 2 (raiz exata = √2)
    f = lambda x: x**2 - 2
    raiz_exata = np.sqrt(2)
    
    history = analise_convergencia_bissecao(f, 1, 2, raiz_exata, max_iter=20)
    
    print("\nEvolução dos erros nas primeiras 10 iterações:")
    print(f"{'Iter':^6} {'Aproximação':^15} {'Erro Abs':^15} {'Erro Rel':^15}")
    print("-" * 55)
    
    for i in range(min(10, len(history['iteracoes']))):
        print(f"{history['iteracoes'][i]:^6} "
              f"{history['aproximacoes'][i]:^15.10f} "
              f"{history['erros_absolutos'][i]:^15.2e} "
              f"{history['erros_relativos'][i]:^15.2e}")
    
    print(f"\nResultado final após {len(history['iteracoes'])} iterações:")
    print(f"  Raiz exata: {raiz_exata:.10f}")
    print(f"  Raiz aproximada: {history['aproximacoes'][-1]:.10f}")
    print(f"  Erro absoluto final: {history['erros_absolutos'][-1]:.2e}")
    print(f"  Erro relativo final: {history['erros_relativos'][-1]:.2e}")
    
    # Plot dos erros (opcional)
    # plot_erros_convergencia(history, "Convergência do Método da Bisseção")
    
    print("\n" + "=" * 80)
    print("EXEMPLO 5: Propagação de Erros")
    print("=" * 80)
    
    a = 10.0
    b = 5.0
    erro_a = 0.1  # 1% de erro
    erro_b = 0.05  # 1% de erro
    
    print(f"Valores: a = {a} ± {erro_a}, b = {b} ± {erro_b}")
    print(f"Erro relativo a: {erro_relativo(erro_a, a)*100:.2f}%")
    print(f"Erro relativo b: {erro_relativo(erro_b, b)*100:.2f}%")
    print()
    
    # Soma
    erro_abs_sum, erro_rel_sum = propagacao_erro_soma(a, b, erro_a, erro_b)
    print(f"Soma: {a+b} ± {erro_abs_sum:.3f} (erro relativo: {erro_rel_sum*100:.3f}%)")
    
    # Produto
    erro_abs_prod, erro_rel_prod = propagacao_erro_produto(a, b, erro_a, erro_b)
    print(f"Produto: {a*b} ± {erro_abs_prod:.3f} (erro relativo: {erro_rel_prod*100:.3f}%)")
    
    # Divisão
    erro_abs_div, erro_rel_div = propagacao_erro_divisao(a, b, erro_a, erro_b)
    print(f"Divisão: {a/b} ± {erro_abs_div:.3f} (erro relativo: {erro_rel_div*100:.3f}%)")
    
    # Plot comparativo (opcional)
    # plot_comparacao_erros(valores_exatos, valores_aprox, 
    #                      labels=['π', 'e', '√2', '1.0'])
    
    print("\n" + "=" * 80)
    print("EXEMPLO 6: Erros em Arrays/Vetores")
    print("=" * 80)
    
    # Trabalhando com arrays
    exato_array = np.array([1, 2, 3, 4, 5])
    aprox_array = np.array([1.01, 1.98, 3.02, 3.97, 5.03])
    
    ea_array = erro_absoluto(aprox_array, exato_array)
    er_array = erro_relativo(aprox_array, exato_array)
    
    print("Valores exatos:   ", exato_array)
    print("Valores aprox:    ", aprox_array)
    print("Erros absolutos:  ", ea_array)
    print("Erros relativos:  ", er_array)
    print("Erros percentuais:", er_array * 100)
    print(f"Erro absoluto médio: {np.mean(ea_array):.4f}")
    print(f"Erro relativo médio: {np.mean(er_array):.4f}")
    
    # Demonstração de precisão numérica
    print("\n" + "=" * 80)
    print("EXEMPLO 7: Comparação de Precisão Numérica")
    print("=" * 80)
    
    # Problema clássico de cancelamento catastrófico
    x = 1e-8
    
    # Fórmula matematicamente equivalente, mas numericamente instável
    formula1 = (1 - np.cos(x)) / (x**2)
    
    # Fórmula estável
    formula2 = (2 * np.sin(x/2)**2) / (x**2)
    
    # Valor exato (usando série de Taylor)
    valor_exato = 0.5 - x**2/24 + x**4/720
    
    print(f"Para x = {x:.2e}")
    print(f"  Fórmula instável: {formula1:.15f} (erro: {erro_absoluto(formula1, valor_exato):.2e})")
    print(f"  Fórmula estável:  {formula2:.15f} (erro: {erro_absoluto(formula2, valor_exato):.2e})")
    print(f"  Valor exato:      {valor_exato:.15f}")