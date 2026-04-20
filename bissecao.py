import numpy as np
import matplotlib.pyplot as plt

def metodo_bissecao(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    Encontra uma raiz da função f(x) no intervalo [a, b] usando o método da bisseção.
    
    Parâmetros:
    f : função contínua (callable)
    a : limite inferior do intervalo
    b : limite superior do intervalo
    tol : tolerância para o critério de parada (default: 1e-6)
    max_iter : número máximo de iterações (default: 100)
    verbose : se True, mostra informações detalhadas das iterações
    
    Retorna:
    x_root : aproximação da raiz
    iterations : número de iterações realizadas
    history : lista com histórico das iterações (opcional)
    
    Lança:
    ValueError: se f(a) e f(b) tiverem o mesmo sinal
    """
    
    # Verifica o teorema de Bolzano
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError(f"O intervalo [{a}, {b}] não contém uma raiz. f({a}) e f({b}) têm o mesmo sinal.")
    elif fa == 0:
        if verbose:
            print(f"Raiz encontrada no limite inferior: x = {a}")
        return a, 0, []
    elif fb == 0:
        if verbose:
            print(f"Raiz encontrada no limite superior: x = {b}")
        return b, 0, []
    
    # Inicialização
    history = []
    iter_count = 0
    
    if verbose:
        print(f"{'Iter':^6} {'a':^12} {'b':^12} {'c':^12} {'f(a)':^12} {'f(b)':^12} {'f(c)':^12} {'Erro':^12}")
        print("-" * 90)
    
    # Loop principal
    for iter_count in range(1, max_iter + 1):
        # Calcula o ponto médio
        c = (a + b) / 2
        fc = f(c)
        
        # Calcula o erro aproximado
        erro = (b - a) / 2
        
        # Armazena histórico
        history.append({
            'iter': iter_count,
            'a': a,
            'b': b,
            'c': c,
            'fa': fa,
            'fb': fb,
            'fc': fc,
            'erro': erro
        })
        
        if verbose:
            print(f"{iter_count:^6} {a:^12.8f} {b:^12.8f} {c:^12.8f} {fa:^12.6e} {fb:^12.6e} {fc:^12.6e} {erro:^12.6e}")
        
        # Critério de parada
        if abs(fc) < tol or erro < tol:
            if verbose:
                print(f"\nConvergência alcançada após {iter_count} iterações")
                print(f"Raiz aproximada: {c:.10f}")
                print(f"Erro aproximado: {erro:.6e}")
                print(f"Valor da função na raiz: {fc:.6e}")
            return c, iter_count, history
        
        # Atualiza o intervalo
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    # Se chegou aqui, não convergiu dentro do número máximo de iterações
    if verbose:
        print(f"\nAtenção: Número máximo de iterações ({max_iter}) atingido sem convergência")
        print(f"Última aproximação: {c:.10f}")
        print(f"Erro: {erro:.6e}")
    
    return c, iter_count, history


def bissecao_simples(f, a, b, tol=1e-6, max_iter=100):
    """
    Versão simplificada do método da bisseção (sem verbose e histórico).
    
    Retorna apenas a raiz aproximada.
    """
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        raise ValueError("O intervalo não contém uma raiz")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return (a + b) / 2


def analise_convergencia(f, a, b, raiz_exata=None):
    """
    Analisa a convergência do método da bisseção.
    
    Parâmetros:
    f : função
    a, b : intervalo inicial
    raiz_exata : valor exato da raiz (se conhecido)
    """
    print("=" * 70)
    print("ANÁLISE DE CONVERGÊNCIA DO MÉTODO DA BISSECÇÃO")
    print("=" * 70)
    
    _, _, history = metodo_bissecao(f, a, b, verbose=False)
    
    print(f"\n{'Iter':^6} {'Erro':^15} {'Erro/Erro_anterior':^20}")
    print("-" * 45)
    
    for i, h in enumerate(history):
        erro = h['erro']
        if i == 0:
            razao = "---"
        else:
            razao = erro / history[i-1]['erro']
        
        print(f"{h['iter']:^6} {erro:^15.6e} {razao:^20.6f}")
    
    print("\nObservação: O método da bisseção tem convergência linear")
    print("com fator de convergência aproximadamente 1/2.")


def visualizar_bissecao(f, a, b, tol=1e-4, max_iter=20):
    """
    Visualiza graficamente o processo do método da bisseção.
    """
    # Executa o método
    raiz, iter_count, history = metodo_bissecao(f, a, b, tol, max_iter, verbose=False)
    
    # Cria o gráfico
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Método da Bisseção - Raiz ≈ {raiz:.6f}', fontsize=14)
    
    # Gráfico 1: Função original com raiz
    x = np.linspace(a - 0.5*(b-a), b + 0.5*(b-a), 1000)
    y = f(x)
    
    axes[0, 0].plot(x, y, 'b-', linewidth=2, label='f(x)')
    axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[0, 0].axvline(x=raiz, color='r', linestyle='--', linewidth=1, label=f'Raiz ≈ {raiz:.4f}')
    axes[0, 0].plot(raiz, 0, 'ro', markersize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].set_title('Função e Raiz Encontrada')
    axes[0, 0].legend()
    
    # Gráfico 2: Convergência dos intervalos
    iter_nums = [h['iter'] for h in history]
    a_vals = [h['a'] for h in history]
    b_vals = [h['b'] for h in history]
    
    for i, (it, a_val, b_val) in enumerate(zip(iter_nums, a_vals, b_vals)):
        axes[0, 1].plot([it, it], [a_val, b_val], 'b-', linewidth=2)
        axes[0, 1].plot(it, a_val, 'ro', markersize=4)
        axes[0, 1].plot(it, b_val, 'ro', markersize=4)
    
    axes[0, 1].axhline(y=raiz, color='r', linestyle='--', linewidth=1, label=f'Raiz = {raiz:.4f}')
    axes[0, 1].set_xlabel('Iteração')
    axes[0, 1].set_ylabel('Intervalo [a, b]')
    axes[0, 1].set_title('Evolução dos Intervalos')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Erro por iteração
    erros = [h['erro'] for h in history]
    
    axes[1, 0].semilogy(iter_nums, erros, 'bo-', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Iteração')
    axes[1, 0].set_ylabel('Erro (log)')
    axes[1, 0].set_title('Convergência do Erro (escala log)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Valor da função no ponto médio
    fc_vals = [h['fc'] for h in history]
    
    axes[1, 1].semilogy(iter_nums, np.abs(fc_vals), 'go-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Iteração')
    axes[1, 1].set_ylabel('|f(c)| (log)')
    axes[1, 1].set_title('Convergência de |f(c)|')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return raiz


# Exemplos de uso e testes
if __name__ == "__main__":
    
    # Exemplo 1: Função polinomial f(x) = x² - 2 (raiz = √2 ≈ 1.4142)
    print("=" * 70)
    print("EXEMPLO 1: f(x) = x² - 2")
    print("=" * 70)
    
    f1 = lambda x: x**2 - 2
    
    raiz1, iter1, hist1 = metodo_bissecao(f1, 1, 2, tol=1e-8, verbose=True)
    print(f"\nResultado: √2 ≈ {raiz1:.10f}")
    print(f"Erro absoluto: {abs(raiz1 - np.sqrt(2)):.2e}")
    
    print("\n" + "=" * 70)
    print("EXEMPLO 2: f(x) = x³ - x - 2")
    print("=" * 70)
    
    # Exemplo 2: f(x) = x³ - x - 2 (raiz ≈ 1.5213797068)
    f2 = lambda x: x**3 - x - 2
    
    raiz2, iter2, hist2 = metodo_bissecao(f2, 1, 2, tol=1e-8, verbose=True)
    print(f"\nResultado: raiz ≈ {raiz2:.10f}")
    
    print("\n" + "=" * 70)
    print("EXEMPLO 3: f(x) = cos(x) - x")
    print("=" * 70)
    
    # Exemplo 3: f(x) = cos(x) - x (raiz ≈ 0.7390851332)
    f3 = lambda x: np.cos(x) - x
    
    raiz3, iter3, hist3 = metodo_bissecao(f3, 0, 1, tol=1e-8, verbose=True)
    print(f"\nResultado: raiz ≈ {raiz3:.10f}")
    
    # Análise de convergência
    print("\n" + "=" * 70)
    analise_convergencia(f1, 1, 2, raiz_exata=np.sqrt(2))
    
    # Versão simplificada
    print("\n" + "=" * 70)
    print("VERSÃO SIMPLIFICADA")
    print("=" * 70)
    
    raiz_simples = bissecao_simples(f1, 1, 2, tol=1e-10)
    print(f"Raiz (versão simples): {raiz_simples:.10f}")
    
    # Visualização gráfica (descomente para ver os gráficos)
    print("\n" + "=" * 70)
    print("GERANDO VISUALIZAÇÃO GRÁFICA...")
    print("=" * 70)
    
    # visualizar_bissecao(f1, 1, 2, tol=1e-6, max_iter=15)
    
    # Exemplo com múltiplas raízes
    print("\n" + "=" * 70)
    print("EXEMPLO 4: f(x) = (x-1)*(x-2)*(x-3) (múltiplas raízes)")
    print("=" * 70)
    
    f4 = lambda x: (x-1)*(x-2)*(x-3)
    
    # Encontrando diferentes raízes
    for intervalo in [(0, 1.5), (1.5, 2.5), (2.5, 4)]:
        try:
            raiz, _, _ = metodo_bissecao(f4, intervalo[0], intervalo[1], tol=1e-8, verbose=False)
            print(f"Raiz no intervalo [{intervalo[0]}, {intervalo[1]}]: {raiz:.10f}")
        except ValueError:
            print(f"Intervalo [{intervalo[0]}, {intervalo[1]}] não contém raiz")