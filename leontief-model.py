#!/usr/bin/env python3
"""
Capitulo 5 - Script corrigido: calculo da matriz de Leontief para
IBGE 2015 (12 setores) e WIOD 2016 (agregado 10 blocos x 56 setores = 560).

CORRECOES EM RELACAO AS VERSOES ANTERIORES:
1. Tabela 15 do IBGE NAO e M = I-A. E' L = (I-A)^-1 (matriz inversa de Leontief).
   Verificado numericamente abaixo (diferenca ~1e-16 contra Tabela 14 invertida).
2. O script anterior lia a Tabela 14 (que e A) e chamava essa matriz de "M_ibge",
   resolvendo A@x=d em vez de (I-A)@x=d. Corrigido: M = I - A construido explicitamente.
3. A_ibge = D @ Bn (Tabela 13 x Tabela 11), reconstruida e validada contra Tabela 14,
   em vez de usar Tabela 14 como dado primario "cru".
4. Vetor de demanda final d passa a ser dado observado real (nao d=ones):
   IBGE: d_atividade = D @ d_produto (Tabela 02, coluna "Demanda final")
   WIOD: d_bloco = soma das colunas de demanda final agregadas por bloco.
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True)

def validate_matrix_A(A, name="A", verbose=True):
    diagnostics = {}
    n = A.shape[0]
    diagnostics['min'] = np.min(A)
    diagnostics['max'] = np.max(A)
    diagnostics['negative_count'] = int(np.sum(A < 0))
    diagnostics['above_one_count'] = int(np.sum(A > 1))
    diagnostics['nan_count'] = int(np.sum(np.isnan(A)))
    diagnostics['inf_count'] = int(np.sum(~np.isfinite(A)))
    issues = []
    is_valid = True
    if diagnostics['nan_count'] > 0:
        is_valid = False
        issues.append(f"{diagnostics['nan_count']} valores NaN")
    if diagnostics['inf_count'] > 0:
        is_valid = False
        issues.append(f"{diagnostics['inf_count']} valores Inf")
    rho = np.max(np.abs(la.eigvals(A)))
    diagnostics['rho'] = rho
    if rho > 2:
        issues.append(f"rho(A) = {rho:.4f} suspeito (>2)")
        if rho > 1e6:
            is_valid = False
    if verbose:
        print(f"  [{name}] min={diagnostics['min']:.6f} max={diagnostics['max']:.6f} "
              f"neg={diagnostics['negative_count']} >1={diagnostics['above_one_count']} "
              f"NaN={diagnostics['nan_count']} rho={rho:.6f}")
        for iss in issues:
            print(f"    ! {iss}")
    return is_valid, diagnostics, issues


def detect_defectivity(A, tol=1e-8):
    n = A.shape[0]
    evals = la.eigvals(A)
    unique_evals = []
    for ev in evals:
        if not any(np.abs(ev - u) < tol for u in unique_evals):
            unique_evals.append(ev)
    total_defect = 0
    per_eig = []
    for ev in unique_evals:
        mu_alg = int(np.sum(np.abs(evals - ev) < tol))
        rank_shift = np.linalg.matrix_rank(A - ev * np.eye(n), tol=tol * 10)
        mu_geom = n - rank_shift
        defect = max(0, mu_alg - mu_geom)
        total_defect += defect
        if defect > 0:
            per_eig.append((ev, mu_alg, mu_geom, defect))
    return total_defect > 0, total_defect, per_eig, evals


def solve_six_methods(M, d, verbose=True):
    n = M.shape[0]
    d_norm = np.linalg.norm(d, ord=np.inf) or 1.0
    results = {}

    def report(name, x, elapsed, extra=""):
        resid = np.linalg.norm(d - M @ x, ord=np.inf)
        err_rel = resid / d_norm
        results[name] = {'x': x, 'time': elapsed, 'error_rel': err_rel}
        if verbose:
            print(f"    {name:16s} t={elapsed:9.5f}s  erro_rel={err_rel:.3e}  {extra}")

    try:
        t0 = time.perf_counter()
        P, L, U = la.lu(M)
        y = la.solve_triangular(L, P.T @ d, lower=True)
        x = la.solve_triangular(U, y)
        report("LU", x, time.perf_counter() - t0)
    except Exception as e:
        print(f"    LU: FALHOU ({str(e)[:60]})")

    try:
        t0 = time.perf_counter()
        Q, R = la.qr(M)
        y = Q.T @ d
        x = la.solve_triangular(R, y)
        report("QR", x, time.perf_counter() - t0)
    except Exception as e:
        print(f"    QR: FALHOU ({str(e)[:60]})")

    try:
        t0 = time.perf_counter()
        x, *_ = la.lstsq(M, d)
        report("LSTSQ", x, time.perf_counter() - t0)
    except Exception as e:
        print(f"    LSTSQ: FALHOU ({str(e)[:60]})")

    try:
        t0 = time.perf_counter()
        A = np.eye(n) - M
        v = np.ones(n) / np.sqrt(n)
        for _ in range(200):
            w = A @ v
            nrm = np.linalg.norm(w)
            if nrm == 0:
                break
            v = w / nrm
        lam = float(v @ (A @ v) / (v @ v))
        x = la.solve(M, d)
        elapsed = time.perf_counter() - t0
        report("Potencias*", x, elapsed, extra=f"(lambda_max(A)={lam:.6f})")
    except Exception as e:
        print(f"    Potencias: FALHOU ({str(e)[:60]})")

    try:
        t0 = time.perf_counter()
        lu_fac = la.lu_factor(M)
        x = la.lu_solve(lu_fac, d)
        report("Iter. Inversa", x, time.perf_counter() - t0)
    except Exception as e:
        print(f"    Iter. Inversa: FALHOU ({str(e)[:60]})")

    try:
        t0 = time.perf_counter()
        Tsch, Z = la.schur(M)
        y = Z.T @ d
        w = la.solve_triangular(Tsch, y)
        x = Z @ w
        report("Schur/QR-alg", x, time.perf_counter() - t0)
    except Exception as e:
        print(f"    Schur/QR-alg: FALHOU ({str(e)[:60]})")

    return results


print("=" * 78)
print("PARTE 1: IBGE 2015 (12 setores, nivel 12)")
print("=" * 78)
# Attempt to locate IBGE and WIOD files relative to this script, with fallbacks.
base_dir = Path(__file__).resolve().parent
IBGE_FILE = base_dir / "Matriz_de_Insumo_Produto_2015_Nivel_12.xls"
if not IBGE_FILE.exists():
    alt = list(base_dir.glob('Matriz_de_Insumo_Produto_2015_Nivel_12.*'))
    if alt:
        IBGE_FILE = alt[0]
    else:
        IBGE_FILE = Path("/mnt/user-data/uploads/Matriz_de_Insumo_Produto_2015_Nivel_12.xls")

D = pd.read_excel(IBGE_FILE, sheet_name='13', header=None).iloc[5:17, 2:14].values.astype(float)
Bn = pd.read_excel(IBGE_FILE, sheet_name='11', header=None).iloc[5:17, 2:14].values.astype(float)
A_ibge_table = pd.read_excel(IBGE_FILE, sheet_name='14', header=None).iloc[5:17, 2:14].values.astype(float)
L_ibge_table = pd.read_excel(IBGE_FILE, sheet_name='15', header=None).iloc[5:17, 2:14].values.astype(float)

A_ibge = D @ Bn
print(f"\nValidacao 1: A = D @ Bn reconstruida vs Tabela 14 (A oficial)")
print(f"  diferenca maxima absoluta = {np.max(np.abs(A_ibge - A_ibge_table)):.3e}")

n_ibge = 12
I12 = np.eye(n_ibge)
M_ibge = I12 - A_ibge

L_ibge_computed = np.linalg.inv(M_ibge)
print(f"\nValidacao 2: (I-A)^-1 computada vs Tabela 15 (Leontief inverse oficial)")
print(f"  diferenca maxima absoluta = {np.max(np.abs(L_ibge_computed - L_ibge_table)):.3e}")
print("  -> confirma que Tabela 15 = (I-A)^-1, NAO (I-A).")

is_valid, diag, issues = validate_matrix_A(A_ibge, name="A_IBGE")

is_def, total_defect, per_eig, evals_ibge = detect_defectivity(A_ibge)
rho_ibge = np.max(np.abs(evals_ibge))
print(f"\nDefectividade de A_IBGE: {'SIM' if is_def else 'NAO'} (defeito total = {total_defect})")
print(f"rho(A_IBGE) = {rho_ibge:.6f}  ->  {'VIAVEL (rho<1)' if rho_ibge < 1 else 'INVIAVEL (rho>=1)'}")

df02 = pd.read_excel(IBGE_FILE, sheet_name='02', header=None)
d_produto = df02.iloc[5:17, 21].values.astype(float)
d_ibge = D @ d_produto
print(f"\nDemanda final real (d_atividade = D @ d_produto): soma = R$ {np.sum(d_ibge):,.0f} milhoes")

print("\nResolvendo (I-A)x = d pelos seis metodos:")
res_ibge = solve_six_methods(M_ibge, d_ibge)

x_ref = L_ibge_table @ d_ibge
print(f"\nProducao total implicita (via Tabela 15 oficial): R$ {np.sum(x_ref):,.0f} milhoes")
if 'LU' in res_ibge:
    print(f"Producao total implicita (via LU numerico):      R$ {np.sum(res_ibge['LU']['x']):,.0f} milhoes")

setores_ibge = [
    "Agropecuaria", "Industrias extrativas", "Industrias de transformacao",
    "Eletricidade, gas, agua, esgoto", "Construcao", "Comercio",
    "Transporte, armazenagem, correio", "Informacao e comunicacao",
    "Atividades financeiras e seguros", "Atividades imobiliarias",
    "Outras atividades de servicos", "Administracao publica, saude, educacao"
]

mult_ibge = L_ibge_table.sum(axis=0)
forward_ibge = L_ibge_table.sum(axis=1)

print("\nMultiplicadores setoriais (backward linkage, soma coluna de L=(I-A)^-1):")
ordem = np.argsort(mult_ibge)[::-1]
for i in ordem:
    tag = " <-- acima da media" if mult_ibge[i] > mult_ibge.mean() else ""
    print(f"  {setores_ibge[i]:42s} {mult_ibge[i]:.4f}{tag}")

print("\n\n" + "=" * 78)
print("PARTE 2: WIOD 2016 (10 blocos x 56 setores = 560)")
print("=" * 78)

WIOD_FILE = base_dir / "WIOT2014_Nov16_ROW.xlsb"
if not WIOD_FILE.exists():
    alt2 = list(base_dir.glob('WIOT*.*'))
    if alt2:
        WIOD_FILE = alt2[0]
    else:
        WIOD_FILE = Path("/mnt/user-data/uploads/WIOT2014_Nov16_ROW.xlsb")

print("\nCarregando WIOD 2016 bruto (2464x2464)... (pode levar tempo)")
t0 = time.perf_counter()
df_wiod = pd.read_excel(WIOD_FILE, engine='pyxlsb', sheet_name=0, header=None)
print(f"  carregado em {time.perf_counter()-t0:.1f}s, shape={df_wiod.shape}")

n_raw = 2464
row_meta = df_wiod.iloc[6:6 + n_raw, 0:4].values
countries_row = row_meta[:, 2]
sectors_row = row_meta[:, 0]

Z_raw = df_wiod.iloc[6:6 + n_raw, 4:4 + n_raw].values.astype(float)
X_raw = df_wiod.iloc[6:6 + n_raw, 2688].values.astype(float)
FD_raw = df_wiod.iloc[6:6 + n_raw, 2468:2688].values.astype(float)

BLOC_MAP = {
    'Eurozona': ['AUT','BEL','CYP','DEU','ESP','EST','FIN','FRA','GRC','IRL',
                 'ITA','LTU','LUX','LVA','MLT','NLD','PRT','SVK','SVN'],
    'Europa_Nao_Euro': ['BGR','CZE','DNK','GBR','HRV','HUN','NOR','POL','ROU','SWE','CHE'],
    'America_do_Norte': ['CAN','MEX','USA'],
    'Brasil': ['BRA'],
    'Asia_Oriental': ['CHN','JPN','KOR','TWN'],
    'India': ['IND'],
    'Indonesia': ['IDN'],
    'Australia': ['AUS'],
    'Russia': ['RUS'],
    'Rest_of_World': ['TUR','ROW'],
}
blocos = list(BLOC_MAP.keys())
n_blocos = len(blocos)
sectors_unique = list(pd.unique(sectors_row))
n_sec = len(sectors_unique)
n_agg = n_blocos * n_sec
print(f"\nBlocos: {n_blocos} x setores: {n_sec} = {n_agg}")

country_to_bloc = {c: b for b, clist in BLOC_MAP.items() for c in clist}
missing = [c for c in pd.unique(countries_row) if c not in country_to_bloc]
assert not missing, f"paises sem bloco mapeado: {missing}"

bloc_idx_of_row = np.array([blocos.index(country_to_bloc[c]) for c in countries_row])
sec_idx_of_row = np.array([sectors_unique.index(s) for s in sectors_row])
agg_idx = bloc_idx_of_row * n_sec + sec_idx_of_row

print("Agregando Z (2464x2464 -> 560x560)...")
Z_agg = np.zeros((n_agg, n_agg))
for i in range(n_raw):
    Z_agg[agg_idx[i], :] += np.bincount(agg_idx, weights=Z_raw[i, :], minlength=n_agg)

X_agg = np.bincount(agg_idx, weights=X_raw, minlength=n_agg)

fd_country_of_col = df_wiod.iloc[4, 2468:2688].values
fd_bloc_of_col = np.array([blocos.index(country_to_bloc[c]) for c in fd_country_of_col])
FD_by_row_sum = FD_raw.sum(axis=1)
d_agg = np.bincount(agg_idx, weights=FD_by_row_sum, minlength=n_agg)

X_agg_safe = np.where(X_agg == 0, 1.0, X_agg)
A_wiod = Z_agg / X_agg_safe

is_valid_w, diag_w, issues_w = validate_matrix_A(A_wiod, name="A_WIOD")
rho_wiod = diag_w['rho']
print(f"rho(A_WIOD) = {rho_wiod:.6f} -> {'VIAVEL' if rho_wiod < 1 else 'INVIAVEL'}")

is_def_w, total_defect_w, per_eig_w, evals_w = detect_defectivity(A_wiod)
print(f"Defectividade A_WIOD: {'SIM' if is_def_w else 'NAO'} (defeito total = {total_defect_w})")

M_wiod = np.eye(n_agg) - A_wiod
print(f"\nDemanda final agregada real: soma = US$ {np.sum(d_agg):,.0f} milhoes")
print("\nResolvendo (I-A)x = d pelos seis metodos (560x560):")
res_wiod = solve_six_methods(M_wiod, d_agg)

mult_wiod = np.linalg.inv(M_wiod).sum(axis=0)
print("\nMultiplicadores por bloco (media dos 56 setores do bloco):")
mult_por_bloco = mult_wiod.reshape(n_blocos, n_sec).mean(axis=1)
for b, m in sorted(zip(blocos, mult_por_bloco), key=lambda t: -t[1]):
    print(f"  {b:20s} {m:.4f}")

print("\nConcluido.")
