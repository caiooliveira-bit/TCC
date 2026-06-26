cat > /mnt/user-data/outputs/README.md << 'ENDOFFILE'
# Modelo Insumo-Produto de Leontief — Implementação Numérica em Python

Estudo do Modelo Insumo-Produto de Wassily Leontief com fundamento em Álgebra Linear e Análise Numérica, implementado em Python com dados reais do IBGE (2015) e da WIOD (2016).

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Estrutura do Repositório](#2-estrutura-do-repositório)
3. [Fluxo do Projeto](#3-fluxo-do-projeto)
4. [Instalação — Linux (Ubuntu/Debian)](#4-instalação--linux-ubuntudebian)
5. [Instalação — macOS](#5-instalação--macos)
6. [Instalação — Windows](#6-instalação--windows)
7. [Configuração das Bases de Dados](#7-configuração-das-bases-de-dados)
8. [Como Executar](#8-como-executar)
9. [Referência de Versões](#9-referência-de-versões)
10. [Métodos Implementados](#10-métodos-implementados)
11. [Solução de Problemas Comuns](#11-solução-de-problemas-comuns)

---

## 1. Visão Geral

O modelo de Leontief descreve as interdependências entre setores produtivos de uma economia por meio de sistemas lineares da forma $(I - A)x = d$, onde $A$ é a matriz de coeficientes técnicos e $d$ o vetor de demanda final.

Este repositório implementa e compara os principais métodos numéricos para resolver e analisar esse sistema, conectando cada algoritmo à teoria de Álgebra Linear que o fundamenta:

| Problema | Método implementado | Fundamento teórico |
|---|---|---|
| Resolver $(I-A)x = d$ | Decomposição LU com pivoteamento | Matrizes triangulares, Determinante |
| Estabilidade numérica | Decomposição QR (Householder) | Matrizes ortogonais, Gram-Schmidt |
| Autovalor dominante $\rho(A)$ | Método das Potências | Diagonalizabilidade |
| Autovalor próximo a $\mu$ | Iteração Inversa | Autoespaço, $(A - \mu I)^{-1}$ |
| Espectro completo de $A$ | Algoritmo QR | Forma de Schur, Cayley-Hamilton |
| Ajuste de séries temporais | Mínimos Quadrados (normais e QR) | Decomposição espectral de $A^TA$ |
| Produtividade da economia | Critério de Hawkins-Simon | Menores principais líderes |

---

## 2. Estrutura do Repositório

```
leontief-numerico/
│
├── README.md                        ← este arquivo
│
├── data/
│   ├── ibge/
│   │   └── MIP_Brasil_2015.xlsx     ← ⚠️ baixar manualmente (ver Seção 7)
│   └── wiod/
│       └── wiot_2016/               ← ⚠️ baixar via pymrio (ver Seção 7)
│           ├── wiot00.xlsx
│           ├── wiot01.xlsx
│           └── ...
│
├── src/
│   ├── main.py                      ← ponto de entrada: carrega dados, executa todos os métodos
│   │
│   ├── model/
│   │   ├── leontief.py              ← construção de A, critério de Hawkins-Simon, série de Neumann
│   │   └── sensitivity.py           ← análise de sensibilidade (perturbações em A e d)
│   │
│   └── methods/
│       ├── lu_decomposition.py      ← fatoração PA = LU com pivoteamento parcial
│       ├── qr_decomposition.py      ← QR via reflexões de Householder
│       ├── power_method.py          ← Método das Potências + Quociente de Rayleigh
│       ├── inverse_iteration.py     ← Iteração Inversa com shift μ
│       ├── qr_algorithm.py          ← Algoritmo QR para espectro completo
│       └── least_squares.py         ← Mínimos Quadrados (equações normais e via QR)
│
├── results/                         ← gerado automaticamente ao rodar main.py
│   ├── *.csv                        ← saídas numéricas (autovalores, soluções, resíduos)
│   └── *.pdf                        ← gráficos (séries temporais WIOD, comparações)
│
├── requirements.txt
└── .gitignore
```

---

## 3. Fluxo do Projeto

```
Dados brutos                  src/                          results/
┌──────────────────┐          ┌──────────────────────────┐  ┌───────────────────────┐
│ IBGE 2015        │          │ leontief.py               │  │                       │
│ MIP_Brasil_2015  │ ──────→  │  Constrói A (68 setores)  │  │  hawkins_simon.csv    │
│ .xlsx            │          │  Critério Hawkins-Simon   │ →│  solucao_lu.csv       │
└──────────────────┘          │  Série de Neumann (I-A)⁻¹ │  │  solucao_qr.csv       │
                              └──────────┬───────────────┘  │  autovalores.csv      │
┌──────────────────┐                     │                   │  series_wiod.pdf      │
│ WIOD 2016        │          ┌──────────▼───────────────┐  │  mq_ajuste.pdf        │
│ 43 países        │ ──────→  │ main.py                   │  │                       │
│ 56 setores       │          │  Carrega dados            │  └───────────────────────┘
│ 2000–2014        │          │  Chama cada método        │
└──────────────────┘          │  Exporta CSVs e PDFs      │
                              └──────────────────────────┘
         ↑
    pymrio carrega
    automaticamente
```

**O que cada módulo faz:**

- `leontief.py` — constrói a matriz $A$ a partir dos dados brutos, verifica o critério de Hawkins-Simon (menores principais líderes) e calcula $(I-A)^{-1}$ via série de Neumann.
- `lu_decomposition.py` — resolve $(I-A)x = d$ por fatoração $PA = LU$; registra o número de condição $\kappa(A)$.
- `qr_decomposition.py` — fatoração $A = QR$ via reflexões de Householder; base para o algoritmo QR e para Mínimos Quadrados.
- `power_method.py` — estima $\rho(A)$ (raio espectral) e seu autovetor; reporta taxa de convergência.
- `inverse_iteration.py` — refina autovalor e autovetor próximos a um shift $\mu$; resolve $(A - \mu I)y = x$ a cada passo via LU.
- `qr_algorithm.py` — computa o espectro completo de $A$ via iterações QR com shifts; verifica invariância por similaridade.
- `least_squares.py` — ajusta tendência temporal às séries WIOD por Mínimos Quadrados via equações normais e via QR; compara resíduos e $\kappa(A^TA)$ vs. $\kappa(A)$.
- `sensitivity.py` — avalia $\|\delta x\| / \|x\|$ para perturbações controladas $\delta A$ e $\delta d$; confronta com o limitante teórico $\kappa(I-A)$.

---

## 4. Instalação — Linux (Ubuntu/Debian)

### 4.1 — Pré-requisitos do sistema

```bash
# Python 3.10 ou superior
python3 --version

# pip atualizado
python3 -m pip install --upgrade pip

# (Opcional) git, caso precise clonar o repositório
sudo apt update && sudo apt install -y git
```

### 4.2 — Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/leontief-numerico.git
cd leontief-numerico
```

### 4.3 — Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 4.4 — Verificar instalação

```bash
python3 -c "import numpy, scipy, pandas, pymrio, matplotlib; print('OK')"
# Saída esperada: OK
```

---

## 5. Instalação — macOS

### 5.1 — Pré-requisitos

Recomenda-se [Homebrew](https://brew.sh) para gerenciar dependências do sistema:

```bash
# Instalar Homebrew (se ainda não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.10+
brew install python@3.11

# Verificar
python3 --version
```

> ⚠️ O Python pré-instalado do macOS (`/usr/bin/python3`) é administrado pelo sistema e pode ter restrições. Use a versão do Homebrew.

### 5.2 — Clonar e instalar

```bash
git clone https://github.com/<seu-usuario>/leontief-numerico.git
cd leontief-numerico

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.3 — Verificar instalação

```bash
python3 -c "import numpy, scipy, pandas, pymrio, matplotlib; print('OK')"
```

---

## 6. Instalação — Windows

### 6.1 — Instalar Python

1. Acesse [python.org/downloads](https://www.python.org/downloads/)
2. Baixe o instalador do **Python 3.11** (ou 3.10+)
3. Execute o instalador e **marque obrigatoriamente** a opção `Add Python to PATH` antes de clicar em "Install Now"
4. Verifique no **PowerShell**:

```powershell
python --version
# Python 3.11.x
```

> ⚠️ Se `python` não for reconhecido após a instalação, feche e reabra o PowerShell. Se ainda falhar, veja a [Seção 11](#11-solução-de-problemas-comuns).

### 6.2 — Clonar e instalar

```powershell
git clone https://github.com/<seu-usuario>/leontief-numerico.git
cd leontief-numerico

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 6.3 — Verificar instalação

```powershell
python -c "import numpy, scipy, pandas, pymrio, matplotlib; print('OK')"
```

---

## 7. Configuração das Bases de Dados

Os dados brutos **não estão versionados** no repositório por questões de tamanho e licença. Siga os passos abaixo para cada base.

### 7.1 — IBGE 2015 (download manual)

1. Acesse: [ibge.gov.br — Matriz de Insumo-Produto](https://www.ibge.gov.br/estatisticas/economicas/contas-nacionais/9085-matriz-de-insumo-produto.html)
2. Baixe o arquivo da **Matriz de Insumo-Produto Brasil 2015** (formato `.xlsx`)
3. Coloque-o em `data/ibge/MIP_Brasil_2015.xlsx`

**Linux / macOS:**
```bash
mkdir -p data/ibge
mv ~/Downloads/MIP_Brasil_2015.xlsx data/ibge/
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path data\ibge
Move-Item "$env:USERPROFILE\Downloads\MIP_Brasil_2015.xlsx" "data\ibge\MIP_Brasil_2015.xlsx"
```

> ℹ️ A matriz IBGE 2015 tem 68 setores e está em valores correntes (R$ milhões).
> O script `leontief.py` seleciona automaticamente o intervalo correto de células.

### 7.2 — WIOD 2016 (download via `pymrio`)

O `pymrio` baixa e organiza a WIOD automaticamente. Execute uma única vez:

```bash
# Com o ambiente virtual ativado
python3 - <<'EOF'
import pymrio
wiod = pymrio.download_wiod(storage_folder="data/wiod/wiot_2016", years=list(range(2000, 2015)))
print("Download concluído.")
EOF
```

> ⚠️ O download é de aproximadamente **800 MB** e pode demorar 10–30 minutos dependendo da conexão.

**O que será baixado:**
```
data/wiod/wiot_2016/
├── wiot00.xlsx    ← tabela world input-output 2000
├── wiot01.xlsx
├── ...
└── wiot14.xlsx    ← tabela world input-output 2014
```

> ℹ️ A WIOD 2016 cobre **43 países**, **56 setores**, anos de **2000 a 2014** (University of Groningen).
> No `main.py`, é possível filtrar por país via a constante `WIOD_COUNTRY = "BRA"`.

---

## 8. Como Executar

Com o ambiente virtual ativado e os dados configurados:

```bash
python3 src/main.py
```

**O que acontece:**

| Etapa | O que faz | Tempo estimado |
|---|---|---|
| 1 — Carregamento | Lê MIP IBGE e séries WIOD | ~30 s (WIOD) |
| 2 — Leontief | Constrói $A$, verifica Hawkins-Simon | < 1 s |
| 3 — LU | Resolve $(I-A)x = d$, calcula $\kappa(I-A)$ | < 1 s |
| 4 — QR | Fatoração de Householder, confere $\|A - QR\|$ | < 1 s |
| 5 — Potências | Estima $\rho(A)$, reporta convergência | < 5 s |
| 6 — It. Inversa | Refina autovalor próximo a $\mu = \rho(A)$ | < 5 s |
| 7 — Alg. QR | Espectro completo, forma de Schur | < 10 s |
| 8 — Mín. Quadrados | Ajuste temporal WIOD, normais vs. QR | < 5 s |
| 9 — Sensibilidade | Perturbações $\delta A$, $\delta d$ | < 2 s |
| 10 — Exportação | Salva CSVs e PDFs em `results/` | < 5 s |

**Saída esperada no terminal (exemplo):**
```
[1/9] Carregando dados...
  ✓ IBGE 2015: matriz 68×68 carregada
  ✓ WIOD 2016: 15 anos, país BRA

[2/9] Critério de Hawkins-Simon...
  ✓ Todos os 68 menores principais líderes positivos
  ✓ ρ(A) = 0.8127 < 1 — economia produtiva

[3/9] Decomposição LU...
  ✓ ||PA - LU|| = 2.31e-14
  ✓ κ(I-A) = 18.43

[4/9] Decomposição QR...
  ✓ ||A - QR|| = 4.17e-15

[5/9] Método das Potências...
  ✓ λ₁ = 0.8127  (convergiu em 47 iterações, |λ₂/λ₁| = 0.961)

...

Resultados salvos em results/
```

---

## 9. Referência de Versões

### Dependências Python

```
# requirements.txt
numpy>=1.26
scipy>=1.12
pandas>=2.2
openpyxl>=3.1
pymrio>=0.5
matplotlib>=3.8
```

### Versões testadas

| Componente | Versão |
|---|---|
| Python | 3.11 (recomendado), 3.10+ (compatível) |
| numpy | 1.26.x |
| scipy | 1.12.x |
| pandas | 2.2.x |
| pymrio | 0.5.x |
| matplotlib | 3.8.x |
| openpyxl | 3.1.x |

### Bases de dados

| Base | Fonte | Período | Cobertura |
|---|---|---|---|
| MIP Brasil | IBGE | 2015 | 68 setores, valores em R$ milhões |
| WIOT | University of Groningen (WIOD 2016) | 2000–2014 | 43 países, 56 setores |

---

## 10. Métodos Implementados

### Decomposição LU — `lu_decomposition.py`

Fatoração $PA = LU$ com pivoteamento parcial. Resolve $Ax = b$ via substituição progressiva ($Lc = Pb$) e regressiva ($Ux = c$). Fundamentada nas propriedades de matrizes triangulares e no Teorema do Determinante.

```
Entrada: A ∈ ℝⁿˣⁿ, b ∈ ℝⁿ
         │
         ├─→ Pivoteamento parcial → P
         ├─→ Eliminação gaussiana → L, U  (PA = LU)
         ├─→ Substituição progressiva: Lc = Pb
         └─→ Substituição regressiva: Ux = c → x
```

### Decomposição QR — `qr_decomposition.py`

Fatoração $A = QR$ via reflexões de Householder (numericamente mais estável que Gram-Schmidt clássico). Base para o Algoritmo QR e para Mínimos Quadrados.

```
Entrada: A ∈ ℝᵐˣⁿ  (m ≥ n)
         │
         └─→ n reflexões de Householder → Q ortogonal, R triangular superior
```

### Método das Potências — `power_method.py`

Calcula o autovalor dominante $\lambda_1 = \rho(A)$ e seu autovetor. Converge quando $|\lambda_1| > |\lambda_2|$; taxa geométrica $|\lambda_2/\lambda_1|^k$.

```
Entrada: A, x⁽⁰⁾ aleatório, tol, max_iter
         │
         └─→ x⁽ᵏ⁺¹⁾ = Ax⁽ᵏ⁾/‖Ax⁽ᵏ⁾‖  até convergência
             └─→ λ₁ via Quociente de Rayleigh: λ⁽ᵏ⁾ = (x⁽ᵏ⁾)ᵀAx⁽ᵏ⁾
```

> ⚠️ Se $|\lambda_2| \approx |\lambda_1|$ (comum em matrizes de Leontief densas), a convergência pode ser lenta. Nesse caso, use o Algoritmo QR.

### Iteração Inversa — `inverse_iteration.py`

Refina autovalor e autovetor próximos a um shift $\mu$. Aplica o Método das Potências a $(A - \mu I)^{-1}$; a resolução em cada passo é feita via LU (fatoração única, reutilizada).

```
Entrada: A, μ (shift), x⁽⁰⁾, tol
         │
         ├─→ Fatoração LU de (A - μI)  [uma vez]
         └─→ Resolve (A - μI)y = x⁽ᵏ⁾  [a cada iteração]
             └─→ converge para autovetor de λⱼ mais próximo de μ
```

### Algoritmo QR — `qr_algorithm.py`

Calcula o espectro completo de $A$ por iterações de similaridade $A_{k+1} = R_k Q_k$. Preserva o polinômio característico a cada passo; converge para a forma de Schur.

```
A₀ = A
│
└─→ A_k = Q_k R_k      (fatoração QR)
    A_{k+1} = R_k Q_k  (similar a A_k, pois A_{k+1} = Qₖᵀ A_k Qₖ)
    │
    └─→ A_k → T triangular superior  (autovalores na diagonal)
```

### Mínimos Quadrados — `least_squares.py`

Ajuste de séries temporais WIOD por polinômio de grau $p$. Implementa e compara dois métodos:

```
Dados: (t₁,y₁), ..., (tₘ,yₘ)  →  min ‖Ax - b‖²
         │
         ├─→ Equações Normais: AᵀAx̂ = Aᵀb    [κ(AᵀA) = κ(A)²]
         └─→ Via QR:           R₁x̂ = Q₁ᵀb    [κ(R₁) = κ(A)]
```

> ℹ️ Para matrizes de Vandermonde mal-condicionadas (graus polinomiais altos), a solução via QR é significativamente mais estável que pelas equações normais.

---

## 11. Solução de Problemas Comuns

### Instalação e ambiente

**`python` não é reconhecido no Windows após a instalação**
→ Reabra o PowerShell após instalar. Se persistir: Painel de Controle → Sistema → Variáveis de Ambiente → adicione o caminho do Python (ex: `C:\Users\<usuario>\AppData\Local\Programs\Python\Python311\`) à variável `Path`.

**`ModuleNotFoundError: No module named 'pymrio'` (ou qualquer outra biblioteca)**
→ Verifique se o ambiente virtual está ativado. O prompt deve mostrar `(.venv)` no início.
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**`pip install` falha com erro de permissão no Linux**
→ Nunca use `sudo pip`. Confirme que está dentro do ambiente virtual antes de instalar.

---

### Base de dados IBGE

**`FileNotFoundError: data/ibge/MIP_Brasil_2015.xlsx`**
→ O arquivo não foi colocado no caminho correto. Verifique o nome exato (sem espaços adicionais). O `leontief.py` procura exatamente por `MIP_Brasil_2015.xlsx`.

**`KeyError` ou `ValueError` ao ler o Excel do IBGE**
→ O IBGE distribui o arquivo com células mescladas. Certifique-se de baixar a versão `.xlsx` (não `.ods` ou `.pdf`). Verifique também se o `openpyxl` está instalado: `pip install openpyxl`.

---

### Base de dados WIOD

**Download via `pymrio` trava ou retorna erro de conexão**
→ O servidor da WIOD pode estar temporariamente indisponível. Tente novamente em alguns minutos, ou baixe manualmente em [wiod.org](http://www.wiod.org/database/wiots16) e coloque os arquivos em `data/wiod/wiot_2016/`.

**`AttributeError: module 'pymrio' has no attribute 'download_wiod'`**
→ Versão antiga do pymrio. Atualize: `pip install --upgrade pymrio`.

---

### Execução dos métodos

**Método das Potências não converge (atinge `max_iter` sem parar)**
→ Provável causa: $|\lambda_1| \approx |\lambda_2|$. Isso ocorre em matrizes de Leontief com setores muito simétricos. Use o Algoritmo QR (`qr_algorithm.py`) para obter o espectro completo sem depender da separação de autovalores.

**Critério de Hawkins-Simon falha (algum menor principal líder ≤ 0)**
→ Indica entradas negativas em $A$ por inconsistências nos dados brutos. Verifique se o intervalo de células lido no Excel está correto — a tabela de fluxos intersetoriais começa em uma linha e coluna específicas; consulte a nota metodológica distribuída pelo IBGE junto ao arquivo.

**`numpy.linalg.LinAlgError: Singular matrix` na Iteração Inversa**
→ O shift $\mu$ coincide com um autovalor exato de $A$. Perturbe levemente: `mu += 1e-10`.

**Resíduos de LU acima de `1e-10`**
→ Verifique se o pivoteamento está ativo (o módulo usa pivoteamento parcial por padrão). Para matrizes com $\kappa > 10^{10}$, o erro de arredondamento em ponto flutuante IEEE 754 é inevitável — o número de condição é reportado na saída do terminal.

**Mínimos Quadrados via equações normais dá resultado diferente da versão QR**
→ Esperado quando $\kappa(A)$ é grande. O quadrado do número de condição ($\kappa(A^TA) = \kappa(A)^2$) amplifica o erro de arredondamento nas equações normais. Para graus polinomiais altos nas séries WIOD, use sempre a versão QR.

---

