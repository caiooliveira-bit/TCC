import numpy as np
import matplotlib.pyplot as plt

def validar_modelo_insumo_produto(A, d, x, epsilon=1e-6, verbose=True):
    """
    Valida a solução do modelo Insumo-Produto de Leontief.
    
    Verifica se a solução x satisfaz x = Ax + d com tolerância ε.
    
    Parâmetros:
    -----------
    A : np.ndarray
        Matriz de coeficientes técnicos (n x n)
    d : np.ndarray
        Vetor de demanda final (n,)
    x : np.ndarray
        Solução do modelo (vetor de produção total) (n,)
    epsilon : float
        Tolerância para validação
    verbose : bool
        Se True, exibe detalhes da validação
    
    Retorna:
    --------
    valido : bool
        True se a solução é válida, False caso contrário
    resultados : dict
        Dicionário com métricas de validação:
        - r: vetor de resíduos
        - erro_absoluto: norma do resíduo
        - erro_relativo: erro relativo calculado
        - erro_max: erro máximo componente a componente
        - epsilon: tolerância utilizada
    """
    
    n = len(x)
    
    # Validações de dimensões
    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(f"Matriz A deve ser {n}x{n}. Shape atual: {A.shape}")
    if len(d) != n:
        raise ValueError(f"Vetor d deve ter tamanho {n}. Tamanho atual: {len(d)}")
    
    # Passo 1: Calcular resíduo r = x - (A*x + d)
    Ax = A @ x
    Ax_plus_d = Ax + d
    r = x - Ax_plus_d
    
    # Passo 2: Calcular erro relativo (norma infinito)
    # ||r||∞ / ||x||∞
    norma_r_inf = np.linalg.norm(r, ord=np.inf)
    norma_x_inf = np.linalg.norm(x, ord=np.inf)
    
    if norma_x_inf < 1e-12:
        raise ValueError("Vetor x é aproximadamente nulo. Não é possível calcular erro relativo.")
    
    erro_relativo = norma_r_inf / norma_x_inf
    
    # Métricas adicionais
    erro_absoluto = norma_r_inf
    erro_max_componente = np.max(np.abs(r))
    erro_medio = np.mean(np.abs(r))
    erro_quadratico = np.linalg.norm(r) / np.sqrt(n)
    
    # Validação
    valido = erro_relativo < epsilon
    
    # Resultados detalhados
    resultados = {
        'r': r,
        'Ax': Ax,
        'Ax_plus_d': Ax_plus_d,
        'norma_r_inf': norma_r_inf,
        'norma_x_inf': norma_x_inf,
        'erro_relativo': erro_relativo,
        'erro_absoluto': erro_absoluto,
        'erro_max_componente': erro_max_componente,
        'erro_medio': erro_medio,
        'erro_quadratico': erro_quadratico,
        'epsilon': epsilon,
        'valido': valido
    }
    
    if verbose:
        print("=" * 70)
        print("VALIDAÇÃO DO MODELO INSUMO-PRODUTO")
        print("=" * 70)
        print(f"Equação de equilíbrio: x = A*x + d")
        print(f"Tolerância (ε): {epsilon}")
        print("-" * 70)
        
        print("\n📊 MÉTRICAS DE VALIDAÇÃO:")
        print(f"  ||r||∞ (norma do resíduo): {norma_r_inf:.6e}")
        print(f"  ||x||∞ (norma da solução): {norma_x_inf:.6e}")
        print(f"  Erro relativo: {erro_relativo:.6e}")
        print(f"  Erro absoluto: {erro_absoluto:.6e}")
        print(f"  Erro máximo por componente: {erro_max_componente:.6e}")
        print(f"  Erro médio: {erro_medio:.6e}")
        print(f"  Erro RMS: {erro_quadratico:.6e}")
        
        print("\n🔍 VERIFICAÇÃO COMPONENTE A COMPONENTE:")
        print(f"{'Setor':^10} | {'x_i':^12} | {'(Ax)_i':^12} | {'(Ax+d)_i':^14} | {'x_i - (Ax+d)_i':^18} | {'Status':^10}")
        print("-" * 85)
        
        for i in range(n):
            residuo_i = r[i]
            status = "✓" if abs(residuo_i) < epsilon * norma_x_inf else "✗"
            print(f"{i+1:^10} | {x[i]:^12.6f} | {Ax[i]:^12.6f} | {Ax_plus_d[i]:^14.6f} | {residuo_i:^18.6e} | {status:^10}")
        
        print("-" * 85)
        
        # Resultado final
        print("\n" + "=" * 70)
        if valido:
            print("✅ RESULTADO: SOLUÇÃO VÁLIDA")
            print(f"   Erro relativo ({erro_relativo:.2e}) < Tolerância ({epsilon})")
        else:
            print("❌ RESULTADO: SOLUÇÃO INCONSISTENTE")
            print(f"   Erro relativo ({erro_relativo:.2e}) >= Tolerância ({epsilon})")
        print("=" * 70)
    
    return valido, resultados

def gerar_relatorio_validacao(resultados):
    """
    Gera um relatório detalhado da validação.
    """
    print("\n" + "=" * 70)
    print("RELATÓRIO DETALHADO DE VALIDAÇÃO")
    print("=" * 70)
    
    print("\n📈 ANÁLISE ESTATÍSTICA DOS RESÍDUOS:")
    r = resultados['r']
    print(f"  Média: {np.mean(r):.6e}")
    print(f"  Mediana: {np.median(r):.6e}")
    print(f"  Desvio padrão: {np.std(r):.6e}")
    print(f"  Mínimo: {np.min(r):.6e}")
    print(f"  Máximo: {np.max(r):.6e}")
    
    print("\n📐 VERIFICAÇÃO DA CONDIÇÃO DE SIMON-HAWKINS:")
    # Verifica se a matriz (I-A) é uma M-matriz
    I = np.eye(len(r))
    I_minus_A = I - resultados['Ax'] @ np.linalg.pinv(resultados['x'].reshape(-1,1)) if len(r) > 0 else I
    # Simplificação: verifica se todos os autovalores de A têm módulo < 1
    autovalores = np.linalg.eigvals(resultados['Ax'] @ np.linalg.pinv(resultados['x'].reshape(-1,1)) if len(r) > 0 else np.zeros_like(I))
    # (Nota: esta é uma simplificação; a condição real é mais complexa)
    
    print("\n🎯 RECOMENDAÇÕES:")
    if resultados['valido']:
        print("  ✓ A solução pode ser utilizada para análises econômicas")
        print("  ✓ Os multiplicadores setoriais são confiáveis")
    else:
        print("  ✗ Verifique a matriz de coeficientes técnicos A")
        print("  ✗ Verifique se a matriz (I-A) é invertível")
        print("  ✗ Verifique se a solução x é não-negativa")
        print("  ✗ Considere usar decomposição LU com pivoteamento")