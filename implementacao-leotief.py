"""
Modelo Insumo-Produto de Leontief
Implementação com dados reais do IBGE - Economia Brasileira 2015
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configuração para salvar imagens em formato vetorial
SAVE_PDF = True      # Salvar como PDF (vetorial, recomendado para LaTeX)
SAVE_SVG = True      # Salvar como SVG (vetorial, para web/edição)
SAVE_PNG = False     # Salvar como PNG (raster, opcional - desligado por padrão)
SAVE_DPI = 300       # Apenas para PNG, se habilitado

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

# Definindo os setores (12 setores da matriz)
SETORES = [
    '01_Agropecuaria',
    '02_Industrias_Extrativas', 
    '03_Industrias_Transformacao',
    '04_Eletricidade_Gas_Agua',
    '05_Construcao',
    '06_Comercio',
    '07_Transporte_Armazenagem',
    '08_Informacao_Comunicacao',
    '09_Financeiro_Seguros',
    '10_Atividades_Imobiliarias',
    '11_Outras_Servicos',
    '12_Administracao_Publica'
]

# Nomes dos setores para visualização (versão mais curta)
SETORES_SHORT = [
    'Agropecuária',
    'Extrativas',
    'Transformação',
    'Utilidades',
    'Construção',
    'Comércio',
    'Transporte',
    'Informação',
    'Financeiro',
    'Imobiliário',
    'Outros Serv.',
    'Administração'
]

# ============================================================================
# 2. MATRIZ DE COEFICIENTES TÉCNICOS (Tabela 14 do IBGE)
# ============================================================================

# Matriz Bn (insumos nacionais) - Tabela 11 do arquivo
A_nacional = np.array([
    [0.04000, 0.00001, 0.07573, 0.00010, 0.00110, 0.00898, 0.00000, 0.00000, 0.00000, 0.00000, 0.00409, 0.00136],
    [0.00086, 0.05488, 0.04234, 0.01485, 0.00984, 0.00006, 0.00001, 0.00000, 0.00000, 0.00058, 0.00004, 0.00006],
    [0.21129, 0.11218, 0.27866, 0.07712, 0.20977, 0.05670, 0.18630, 0.02682, 0.00874, 0.00933, 0.07127, 0.02569],
    [0.02349, 0.01086, 0.01556, 0.27802, 0.00115, 0.01746, 0.00572, 0.00718, 0.00405, 0.00125, 0.01664, 0.01736],
    [0.00060, 0.01281, 0.00091, 0.01302, 0.09564, 0.00094, 0.00317, 0.01665, 0.00298, 0.00298, 0.00356, 0.01420],
    [0.05568, 0.02765, 0.07331, 0.01811, 0.05383, 0.02541, 0.04239, 0.02296, 0.00451, 0.00301, 0.02988, 0.01201],
    [0.02034, 0.08462, 0.04987, 0.01852, 0.01193, 0.05042, 0.11418, 0.00857, 0.01358, 0.00079, 0.01780, 0.01133],
    [0.00009, 0.00377, 0.00549, 0.00651, 0.00214, 0.01302, 0.00755, 0.12201, 0.03955, 0.00141, 0.03820, 0.01741],
    [0.01526, 0.02210, 0.01747, 0.02156, 0.01423, 0.02297, 0.02460, 0.02843, 0.12377, 0.03788, 0.01653, 0.04445],
    [0.00004, 0.00148, 0.00200, 0.00469, 0.00181, 0.03696, 0.00736, 0.01337, 0.01051, 0.00300, 0.01913, 0.00394],
    [0.00361, 0.09914, 0.04852, 0.05392, 0.02456, 0.08481, 0.06124, 0.14994, 0.10861, 0.00849, 0.09614, 0.08346],
    [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]
])

# ============================================================================
# 3. VETOR DE PRODUÇÃO TOTAL
# ============================================================================

producao_total = np.array([
    465342, 251737, 2802997, 321797, 644583, 1037004, 499268, 349059, 564015, 596597, 1558276, 1136194
])

# ============================================================================
# 4. VETOR DE DEMANDA FINAL
# ============================================================================

demanda_final = np.array([
    217656, 107905, 1429422, 100770, 539568, 639588, 151174, 167754, 269009, 498332, 876965, 1136194
])

# ============================================================================
# 5. FUNÇÕES DO MODELO INSUMO-PRODUTO
# ============================================================================

def verificar_matriz_produtiva(A):
    soma_colunas = np.sum(A, axis=0)
    return np.all(soma_colunas < 1), soma_colunas

def matriz_leontief(A):
    n = len(A)
    I = np.eye(n)
    return I - A

def calcular_multiplicadores(A):
    L = matriz_leontief(A)
    try:
        multiplicadores = np.linalg.inv(L)
        return multiplicadores
    except np.linalg.LinAlgError:
        print("Erro: Matriz singular - não é possível inverter")
        return None

def resolver_modelo(A, d):
    multiplicadores = calcular_multiplicadores(A)
    if multiplicadores is not None:
        x = multiplicadores @ d
        return x, multiplicadores
    return None, None

def calcular_consumo_intermediario(A, x):
    return A @ x

def verificar_consistencia(x, A, d, tolerancia=1e-6):
    residuo = x - (A @ x + d)
    erro_relativo = np.linalg.norm(residuo) / np.linalg.norm(x)
    return erro_relativo < tolerancia, erro_relativo

# ============================================================================
# 6. FUNÇÃO AUXILIAR PARA SALVAR IMAGENS VETORIAIS
# ============================================================================

def salvar_imagem(nome_base):
    """
    Salva a figura atual nos formatos especificados (PDF, SVG e opcionalmente PNG)
    """
    if SAVE_PDF:
        plt.savefig(f'{nome_base}.pdf', format='pdf', bbox_inches='tight')
    if SAVE_SVG:
        plt.savefig(f'{nome_base}.svg', format='svg', bbox_inches='tight')
    if SAVE_PNG:
        plt.savefig(f'{nome_base}.png', dpi=SAVE_DPI, bbox_inches='tight')
    
    # Mensagem de feedback
    formatos = []
    if SAVE_PDF: formatos.append('PDF')
    if SAVE_SVG: formatos.append('SVG')
    if SAVE_PNG: formatos.append('PNG')
    print(f"   ✓ {nome_base}.{'/'.join(formatos)} salvo")

# ============================================================================
# 7. EXECUÇÃO DO MODELO
# ============================================================================

print("=" * 80)
print("MODELO INSUMO-PRODUTO DE LEONTIEF")
print("Economia Brasileira - Dados 2015 (IBGE)")
print("=" * 80)

produtiva, soma_colunas = verificar_matriz_produtiva(A_nacional)
print(f"\n1. Verificação da Matriz de Coeficientes Técnicos:")
print(f"   - Matriz é produtiva: {produtiva}")

L = matriz_leontief(A_nacional)
x_calculado, multiplicadores = resolver_modelo(A_nacional, demanda_final)

if x_calculado is not None:
    erro_medio = np.mean(np.abs(x_calculado - producao_total) / producao_total * 100)
    consistente, erro_rel = verificar_consistencia(x_calculado, A_nacional, demanda_final)
    print(f"\n   Erro médio absoluto: {erro_medio:.4f}%")
    print(f"   Solução consistente: {consistente}")

# ============================================================================
# 8. FUNÇÕES PARA GERAR GRÁFICOS VETORIAIS INDIVIDUAIS
# ============================================================================

def grafico_producao_setores():
    """Gráfico 1: Produção Total por Setor"""
    plt.figure(figsize=(12, 8))
    cores = plt.cm.viridis(np.linspace(0, 0.9, len(SETORES_SHORT)))
    barras = plt.barh(SETORES_SHORT, producao_total / 1e6, color=cores)
    plt.xlabel('Produção (R$ bilhões)', fontsize=12)
    plt.ylabel('Setor', fontsize=12)
    plt.title('Produção Total por Setor - Brasil 2015', fontsize=14, fontweight='bold')
    
    for barra, valor in zip(barras, producao_total / 1e6):
        plt.text(valor + 10, barra.get_y() + barra.get_height()/2, 
                f'R$ {valor:.1f}B', va='center', fontsize=9)
    
    plt.tight_layout()
    salvar_imagem('grafico_01_producao_setores')
    plt.close()

def grafico_matriz_coeficientes():
    """Gráfico 2: Mapa de Calor da Matriz de Coeficientes Técnicos"""
    plt.figure(figsize=(14, 10))
    im = plt.imshow(A_nacional, cmap='YlOrRd', aspect='auto')
    plt.xticks(range(len(SETORES_SHORT)), SETORES_SHORT, rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(SETORES_SHORT)), SETORES_SHORT, fontsize=9)
    plt.xlabel('Setor de Destino (insumo para...)', fontsize=12)
    plt.ylabel('Setor de Origem (produto de...)', fontsize=12)
    plt.title('Matriz de Coeficientes Técnicos (A)', fontsize=14, fontweight='bold')
    
    # Adicionar valores nas células (apenas para valores significativos)
    for i in range(len(SETORES_SHORT)):
        for j in range(len(SETORES_SHORT)):
            if A_nacional[i, j] > 0.01:
                plt.text(j, i, f'{A_nacional[i, j]:.3f}', ha='center', va='center', fontsize=7, 
                        color='white' if A_nacional[i, j] > 0.15 else 'black')
    
    plt.colorbar(im, label='Coeficiente')
    plt.tight_layout()
    salvar_imagem('grafico_02_matriz_coeficientes')
    plt.close()

def grafico_dependencia_insumos():
    """Gráfico 3: Dependência de Insumos por Setor"""
    plt.figure(figsize=(12, 8))
    soma_colunas = np.sum(A_nacional, axis=0)
    cores = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(SETORES_SHORT)))
    barras = plt.barh(SETORES_SHORT, soma_colunas, color=cores)
    plt.xlabel('Soma dos Coeficientes (Σ a_ij)', fontsize=12)
    plt.ylabel('Setor', fontsize=12)
    plt.title('Dependência de Insumos por Setor', fontsize=14, fontweight='bold')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Limite (sum=1)')
    plt.legend(loc='lower right')
    
    for barra, valor in zip(barras, soma_colunas):
        plt.text(valor + 0.02, barra.get_y() + barra.get_height()/2, 
                f'{valor:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    salvar_imagem('grafico_03_dependencia_insumos')
    plt.close()

def grafico_validacao_modelo():
    """Gráfico 4: Validação do Modelo - Produção Real vs Calculada"""
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(SETORES_SHORT))
    width = 0.35
    
    plt.barh(x_pos - width/2, producao_total / 1e6, width, 
             label='Produção Real (IBGE)', color='steelblue', alpha=0.8)
    plt.barh(x_pos + width/2, x_calculado / 1e6, width, 
             label='Produção Calculada', color='coral', alpha=0.8)
    
    plt.yticks(x_pos, SETORES_SHORT)
    plt.xlabel('Produção (R$ bilhões)', fontsize=12)
    plt.title('Validação do Modelo: Produção Real vs Calculada', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    salvar_imagem('grafico_04_validacao_modelo')
    plt.close()

def grafico_erro_relativo():
    """Gráfico 5: Erro Relativo por Setor"""
    plt.figure(figsize=(12, 8))
    erros = np.abs(x_calculado - producao_total) / producao_total * 100
    cores = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(SETORES_SHORT)))[::-1]
    barras = plt.barh(SETORES_SHORT, erros, color=cores)
    plt.xlabel('Erro Relativo (%)', fontsize=12)
    plt.ylabel('Setor', fontsize=12)
    plt.title('Erro Relativo da Solução do Modelo por Setor', fontsize=14, fontweight='bold')
    plt.axvline(x=erro_medio, color='red', linestyle='--', linewidth=2, label=f'Média: {erro_medio:.2f}%')
    plt.legend(loc='lower right')
    
    for barra, erro in zip(barras, erros):
        plt.text(erro + 0.5, barra.get_y() + barra.get_height()/2, 
                f'{erro:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    salvar_imagem('grafico_05_erro_relativo')
    plt.close()

def grafico_multiplicadores():
    """Gráfico 6: Principais Multiplicadores (Efeitos Próprios)"""
    plt.figure(figsize=(12, 8))
    efeitos_proprios = np.diag(multiplicadores)
    cores = plt.cm.Blues(np.linspace(0.3, 0.9, len(SETORES_SHORT)))
    barras = plt.barh(SETORES_SHORT, efeitos_proprios, color=cores)
    plt.xlabel('Multiplicador (efeito sobre si mesmo)', fontsize=12)
    plt.ylabel('Setor', fontsize=12)
    plt.title('Multiplicadores de Leontief - Efeitos Totais por Setor', fontsize=14, fontweight='bold')
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Referência (valor 1)')
    plt.legend(loc='lower right')
    
    for barra, valor in zip(barras, efeitos_proprios):
        plt.text(valor + 0.05, barra.get_y() + barra.get_height()/2, 
                f'{valor:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    salvar_imagem('grafico_06_multiplicadores')
    plt.close()

def grafico_valor_adicionado():
    """Gráfico 7: Valor Adicionado por Setor"""
    consumo_intermediario = calcular_consumo_intermediario(A_nacional, producao_total)
    valor_adicionado = producao_total - consumo_intermediario
    
    plt.figure(figsize=(12, 8))
    indices_ordenados = np.argsort(valor_adicionado)[::-1]
    setores_ordenados = [SETORES_SHORT[i] for i in indices_ordenados]
    va_ordenado = [valor_adicionado[i] / 1e6 for i in indices_ordenados]
    
    cores = plt.cm.Greens(np.linspace(0.3, 0.9, len(setores_ordenados)))
    barras = plt.barh(setores_ordenados, va_ordenado, color=cores)
    plt.xlabel('Valor Adicionado (R$ milhões)', fontsize=12)
    plt.ylabel('Setor', fontsize=12)
    plt.title('Valor Adicionado por Setor - Economia Brasileira 2015', fontsize=14, fontweight='bold')
    
    total_va = np.sum(valor_adicionado)
    for barra, valor in zip(barras, va_ordenado):
        participacao = valor / (total_va / 1e6) * 100
        plt.text(valor + 5000, barra.get_y() + barra.get_height()/2, 
                f'R$ {valor:.0f}M ({participacao:.1f}%)', va='center', fontsize=9)
    
    plt.tight_layout()
    salvar_imagem('grafico_07_valor_adicionado')
    plt.close()

def grafico_analise_sensibilidade():
    """Gráfico 8: Análise de Sensibilidade do Modelo"""
    plt.figure(figsize=(10, 6))
    perturbacoes = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    sensibilidades = []
    
    x_original = np.linalg.solve(np.eye(len(A_nacional)) - A_nacional, demanda_final)
    
    for p in perturbacoes:
        sensibilidades_amostra = []
        for _ in range(10):  # 10 amostras por nível de perturbação
            d_perturbado = demanda_final * (1 + p * np.random.randn(len(demanda_final)))
            x_perturbado = np.linalg.solve(np.eye(len(A_nacional)) - A_nacional, d_perturbado)
            sens = np.linalg.norm(x_perturbado - x_original) / np.linalg.norm(x_original)
            sensibilidades_amostra.append(sens)
        sensibilidades.append(np.mean(sensibilidades_amostra))
    
    plt.plot(perturbacoes * 100, sensibilidades * 100, 'o-', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Perturbação na Demanda Final (%)', fontsize=12)
    plt.ylabel('Variação na Produção Total (%)', fontsize=12)
    plt.title('Análise de Sensibilidade do Modelo Insumo-Produto', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Adicionar pontos com valores
    for p, s in zip(perturbacoes * 100, sensibilidades * 100):
        plt.annotate(f'{s:.2f}%', (p, s), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)
    
    plt.tight_layout()
    salvar_imagem('grafico_08_analise_sensibilidade')
    plt.close()

def grafico_participacao_setorial():
    """Gráfico 9: Participação Percentual de Cada Setor na Produção Total"""
    plt.figure(figsize=(10, 10))
    participacao = producao_total / np.sum(producao_total) * 100
    cores = plt.cm.Set3(np.linspace(0, 1, len(SETORES_SHORT)))
    
    plt.pie(participacao, labels=SETORES_SHORT, autopct='%1.1f%%', startangle=90, colors=cores)
    plt.title('Participação Setorial na Produção Total - Brasil 2015', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    salvar_imagem('grafico_09_participacao_setorial')
    plt.close()

def grafico_comparacao_multiplicadores():
    """Gráfico 10: Comparação dos Multiplicadores (Top 5 setores)"""
    plt.figure(figsize=(12, 8))
    efeitos_proprios = np.diag(multiplicadores)
    indices_top5 = np.argsort(efeitos_proprios)[-5:]
    
    x_pos = np.arange(5)
    width = 0.35
    
    top_setores = [SETORES_SHORT[i] for i in indices_top5]
    top_efeitos = efeitos_proprios[indices_top5]
    
    plt.bar(x_pos, top_efeitos, color='steelblue', alpha=0.8)
    plt.xticks(x_pos, top_setores, rotation=45, ha='right')
    plt.ylabel('Multiplicador (efeito total)', fontsize=12)
    plt.title('Top 5 Setores com Maiores Multiplicadores de Leontief', fontsize=14, fontweight='bold')
    
    for i, (setor, valor) in enumerate(zip(top_setores, top_efeitos)):
        plt.text(i, valor + 0.02, f'{valor:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    salvar_imagem('grafico_10_comparacao_multiplicadores')
    plt.close()

# ============================================================================
# 9. EXECUTAR TODOS OS GRÁFICOS
# ============================================================================

print("\n2. Gerando visualizações vetoriais...")
print("   " + "-" * 50)
print(f"   Formatos: {'PDF, ' if SAVE_PDF else ''}{'SVG, ' if SAVE_SVG else ''}{'PNG' if SAVE_PNG else ''}")
print("   " + "-" * 50)

grafico_producao_setores()
grafico_matriz_coeficientes()
grafico_dependencia_insumos()
grafico_validacao_modelo()
grafico_erro_relativo()
grafico_multiplicadores()
grafico_valor_adicionado()
grafico_analise_sensibilidade()
grafico_participacao_setorial()
grafico_comparacao_multiplicadores()

print("   " + "-" * 50)
print("   ✓ Todos os 10 gráficos foram gerados em formato vetorial!")

# ============================================================================
# 10. CÁLCULO DO VETOR DE VALOR ADICIONADO (para o relatório)
# ============================================================================

consumo_intermediario = calcular_consumo_intermediario(A_nacional, producao_total)
valor_adicionado = producao_total - consumo_intermediario
total_va = np.sum(valor_adicionado)

# ============================================================================
# 11. RELATÓRIO FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RELATÓRIO FINAL - MODELO INSUMO-PRODUTO")
print("=" * 80)

print(f"""
RESUMO DOS RESULTADOS:

1. CARACTERÍSTICAS DO MODELO:
   - Número de setores: {len(SETORES)}
   - Matriz de coeficientes: {A_nacional.shape[0]}x{A_nacional.shape[1]}
   - Determinante de (I-A): {np.linalg.det(L):.6e}
   - Matriz é produtiva: {produtiva}

2. PRINCIPAIS MULTIPLICADORES (Efeitos totais):
""")

for i in range(min(5, len(SETORES_SHORT))):
    print(f"   - {SETORES_SHORT[i]:25s}: {multiplicadores[i, i]:.4f}")

print(f"""
3. PRINCIPAIS DEPENDÊNCIAS SETORIAIS:
   - Setor que mais depende de insumos: {SETORES_SHORT[np.argmax(soma_colunas)]}
   - Setor que menos depende de insumos: {SETORES_SHORT[np.argmin(soma_colunas)]}

4. VALOR ADICIONADO TOTAL: R$ {total_va/1e6:.1f} milhões

5. PRECISÃO DO MODELO:
   - Erro médio da solução: {erro_medio:.4f}%
   - Consistência do modelo: {consistente}

6. GRÁFICOS VETORIAIS GERADOS:
""")

arquivos = [
    "grafico_01_producao_setores",
    "grafico_02_matriz_coeficientes", 
    "grafico_03_dependencia_insumos",
    "grafico_04_validacao_modelo",
    "grafico_05_erro_relativo",
    "grafico_06_multiplicadores",
    "grafico_07_valor_adicionado",
    "grafico_08_analise_sensibilidade",
    "grafico_09_participacao_setorial",
    "grafico_10_comparacao_multiplicadores"
]

for arquivo in arquivos:
    extensoes = []
    if SAVE_PDF: extensoes.append('pdf')
    if SAVE_SVG: extensoes.append('svg')
    if SAVE_PNG: extensoes.append('png')
    print(f"   - {arquivo}.{', '.join(extensoes)}")

print("""
7. COMO USAR AS IMAGENS VETORIAIS NO LaTeX:
   
   Para PDF (recomendado):
   \\usepackage{graphicx}
   \\includegraphics[width=\\textwidth]{grafico_01_producao_setores.pdf}
   
   Para SVG (requer pacote svg):
   \\usepackage{svg}
   \\includesvg{grafico_01_producao_setores.svg}
""")

print("=" * 80)
print("Análise concluída com sucesso!")
print("=" * 80)