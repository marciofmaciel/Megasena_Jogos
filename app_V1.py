import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

st.set_page_config(page_title="Mega-Sena Painel 3.0", layout="wide")
st.title("🎯 Painel Inteligente de Previsão da Mega-Sena – Versão 3.0")

uploaded_file = st.file_uploader("📂 Carregar arquivo Excel", type=["xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file, header=0)
dezenas_df = df.iloc[1:, 2:8].reset_index(drop=True)

# ================================================================
# Funções auxiliares
# ================================================================

TODAS_DEZENAS = np.arange(1, 61)

def _linha_valida(row):
    return set(int(x) for x in row if pd.notna(x) and 1 <= float(x) <= 60)

# ================================================================
# 🌀 TEORIA DO CAOS — Funções de análise de complexidade
# ================================================================

def _serie_temporal_dezena(dezenas_df, dezena):
    """
    Série temporal binária: 1 se a dezena saiu no concurso, 0 se não.
    Base de dados para todas as métricas de caos.
    """
    return np.array([
        1 if dezena in _linha_valida(dezenas_df.iloc[i]) else 0
        for i in range(len(dezenas_df))
    ], dtype=float)


def calcular_hurst(serie):
    """
    Expoente de Hurst via método R/S.
    H > 0.55 -> persistente  | H < 0.45 -> anti-persistente | H ≈ 0.5 -> aleatório
    """
    n = len(serie)
    if n < 20:
        return 0.5
    media  = np.mean(serie)
    desvio = serie - media
    cumsum = np.cumsum(desvio)
    R = np.max(cumsum) - np.min(cumsum)
    S = np.std(serie, ddof=1)
    if S == 0 or R == 0:
        return 0.5
    return float(np.log(R / S) / np.log(n))


def calcular_lyapunov(serie):
    """
    Expoente de Lyapunov aproximado.
    λ > 0 -> caótico (imprevisível) | λ < 0 -> estável/convergente
    """
    n = len(serie)
    if n < 10:
        return 0.0
    diffs = np.abs(np.diff(serie))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return float(np.mean(np.log(diffs + 1e-10)))


def calcular_entropia_permutacao(serie, ordem=3):
    """
    Entropia de Permutação normalizada.
    E → 0 = série regular/previsível | E → 1 = série caótica/desordenada
    """
    from math import factorial
    n = len(serie)
    if n < ordem + 1:
        return 1.0
    contagem = {}
    total = 0
    for i in range(n - ordem):
        padrao = tuple(np.argsort(serie[i:i + ordem]))
        contagem[padrao] = contagem.get(padrao, 0) + 1
        total += 1
    entropia = sum(-(c/total) * np.log(c/total) for c in contagem.values())
    max_entropia = np.log(factorial(ordem))
    return float(entropia / max_entropia) if max_entropia > 0 else 1.0


def calcular_regime_caos(hurst, lyapunov, entropia):
    """
    Classifica o regime dinâmico e retorna (label, score_oportunidade).
    Score alto = regime favorável para previsão.
    """
    if lyapunov > 0.0 and entropia > 0.75:
        return "⚠️ Caótico",         round(max(0.1, 1.0 - entropia), 4)
    elif hurst > 0.55:
        return "📈 Persistente",      round(min(1.0, hurst * 1.2), 4)
    elif hurst < 0.45:
        return "🔄 Anti-persistente", round(0.5 + (0.5 - hurst), 4)
    else:
        return "🌊 Estocástico",      0.5


@st.cache_data
def calcular_metricas_caos(dezenas_df):
    """
    Calcula Hurst, Lyapunov e Entropia de Permutação para cada dezena.
    Retorna dict {dezena: {hurst, lyapunov, entropia, regime, oportunidade, chaos_score}}
    """
    resultado = {}
    for n in TODAS_DEZENAS:
        serie = _serie_temporal_dezena(dezenas_df, int(n))
        h = calcular_hurst(serie)
        l = calcular_lyapunov(serie)
        e = calcular_entropia_permutacao(serie)
        regime, oportunidade = calcular_regime_caos(h, l, e)

        # ChaosScore [0,1]: bonifica persistência e baixa entropia, penaliza caos
        chaos_score = (
            np.clip(h, 0, 1) * 0.5 +
            (1.0 - np.clip(e, 0, 1)) * 0.3 +
            (1.0 - np.clip((l + 2) / 4, 0, 1)) * 0.2
        )
        resultado[int(n)] = {
            "hurst":        round(h, 4),
            "lyapunov":     round(l, 4),
            "entropia":     round(e, 4),
            "regime":       regime,
            "oportunidade": oportunidade,
            "chaos_score":  round(float(chaos_score), 4),
        }
    return resultado


# ================================================================
# Estatísticas principais + score integrado com caos
# ================================================================

@st.cache_data
def calcular_estatisticas(dezenas_df, pesos):
    ult50 = dezenas_df.tail(50)
    ult20 = dezenas_df.tail(20)

    vals       = dezenas_df.values.flatten()
    vals_valid = vals[pd.notna(vals)]
    freq_total = {int(n): int((vals_valid == n).sum()) for n in TODAS_DEZENAS}
    freq50     = {int(n): int((ult50.values.flatten()[pd.notna(ult50.values.flatten())] == n).sum()) for n in TODAS_DEZENAS}
    freq20     = {int(n): int((ult20.values.flatten()[pd.notna(ult20.values.flatten())] == n).sum()) for n in TODAS_DEZENAS}

    atraso = {}
    for n in TODAS_DEZENAS:
        linhas = np.where((dezenas_df == n).any(axis=1))[0]
        atraso[int(n)] = len(dezenas_df) - int(linhas.max()) if len(linhas) > 0 else len(dezenas_df)

    repeticoes = {int(n): 0 for n in TODAS_DEZENAS}
    for i in range(1, len(dezenas_df)):
        anterior = _linha_valida(dezenas_df.iloc[i-1])
        atual    = _linha_valida(dezenas_df.iloc[i])
        for n in anterior.intersection(atual):
            repeticoes[int(n)] += 1
    repeticoes = {n: repeticoes[n] / len(dezenas_df) for n in repeticoes}

    afinidade = {int(n): freq_total[int(n)] for n in TODAS_DEZENAS}

    # 🌀 Métricas de caos
    caos = calcular_metricas_caos(dezenas_df)

    def norm(d):
        m = max(d.values())
        return {k: d[k] / m if m > 0 else 0 for k in d}

    f50 = norm(freq50)
    f20 = norm(freq20)
    a   = norm(atraso)
    r   = norm(repeticoes)
    c   = norm(afinidade)
    ch  = norm({n: caos[n]["chaos_score"] for n in caos})

    w50, w20, wA, wR, wC, wCH = pesos

    score = {}
    for n in TODAS_DEZENAS:
        ni = int(n)
        score[ni] = (
            w50 * f50[ni] +
            w20 * f20[ni] +
            wA  * a[ni]   +
            wR  * r[ni]   +
            wC  * c[ni]   +
            wCH * ch[ni]   # 🌀 peso do caos
        )

    total_score = sum(score.values())
    prob = {n: score[n] / total_score for n in score} if total_score > 0 else {n: 1/60 for n in score}

    return freq50, freq20, atraso, repeticoes, afinidade, caos, score, prob


# ================================================================
# Filtros de geração de jogos
# ================================================================

def quadrantes_ok(comb):
    q = set()
    for n in comb:
        if   1  <= n <= 15: q.add(1)
        elif 16 <= n <= 30: q.add(2)
        elif 31 <= n <= 45: q.add(3)
        else:               q.add(4)
    return len(q) >= 3

def faixas_ok(comb):
    return len(set((n - 1) // 10 for n in comb)) >= 3

def sequencias_ok(comb, max_seq=2):
    comb_ord = sorted(comb)
    seq = 1
    for i in range(1, len(comb_ord)):
        if comb_ord[i] == comb_ord[i-1] + 1:
            seq += 1
            if seq > max_seq:
                return False
        else:
            seq = 1
    return True

def recentes_ok(comb, dezenas_df, max_rec=2, ult_n=3):
    recentes = _linha_valida(dezenas_df.tail(ult_n).values.flatten())
    return sum(1 for n in comb if n in recentes) <= max_rec

def passa_filtros(comb, dezenas_df, min_pares, max_pares, soma_min, soma_max):
    pares = sum(1 for x in comb if x % 2 == 0)
    soma  = sum(comb)
    return (
        min_pares <= pares <= max_pares and
        soma_min  <= soma  <= soma_max  and
        quadrantes_ok(comb)             and
        faixas_ok(comb)                 and
        sequencias_ok(comb, max_seq=2)  and
        recentes_ok(comb, dezenas_df)
    )

def gerar_jogos(prob, dezenas_df, qtd_jogos, num_sim, filtros, evitar_repeticao=True):
    resultados = {}
    p_array = np.array([prob[int(n)] for n in TODAS_DEZENAS])
    for _ in range(num_sim):
        sorteio = np.random.choice(TODAS_DEZENAS, size=6, replace=False, p=p_array)
        sorteio = tuple(sorted(int(x) for x in sorteio))
        resultados[sorteio] = resultados.get(sorteio, 0) + 1

    ordenado = sorted(resultados.items(), key=lambda x: x[1], reverse=True)
    min_pares, max_pares, soma_min, soma_max = filtros
    jogos, usadas = [], set()

    for comb, _ in ordenado:
        if len(jogos) >= qtd_jogos:
            break
        if not passa_filtros(comb, dezenas_df, min_pares, max_pares, soma_min, soma_max):
            continue
        if evitar_repeticao and any(n in usadas for n in comb):
            continue
        jogos.append(comb)
        usadas.update(comb)
    return jogos

def backtest(dezenas_df, pesos, num_sim, jogos_por_concurso, inicio, fim, filtros):
    resultados_bt = []
    for idx in range(inicio, fim):
        historico = dezenas_df.iloc[:idx]
        real = _linha_valida(dezenas_df.iloc[idx])
        *_, prob = calcular_estatisticas(historico, pesos)
        jogos = gerar_jogos(prob, historico, jogos_por_concurso, num_sim, filtros, evitar_repeticao=False)
        for jogo in jogos:
            acertos = len(real.intersection(set(jogo)))
            resultados_bt.append({"concurso": idx+1, "jogo": jogo, "acertos": acertos})
    return pd.DataFrame(resultados_bt)


# ================================================================
# Abas
# ================================================================

tab_modelo, tab_dezenas, tab_caos, tab_jogos, tab_pacotes, tab_backtest = st.tabs([
    "Modelo & Pesos",
    "Análise de Dezenas",
    "🌀 Teoria do Caos",
    "Jogos Otimizados",
    "Pacotes de Perfis",
    "Backtest & Métricas"
])

# ----------------------------------------------------------------
# Aba: Modelo & Pesos
# ----------------------------------------------------------------
with tab_modelo:
    st.subheader("⚙ Ajuste de Pesos do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        w50 = st.slider("Peso frequência 50",      0.0, 1.0, 0.35, 0.05)
        w20 = st.slider("Peso frequência 20",      0.0, 1.0, 0.20, 0.05)
        wA  = st.slider("Peso atraso",             0.0, 1.0, 0.15, 0.05)
    with col2:
        wR  = st.slider("Peso repetição",          0.0, 1.0, 0.10, 0.05)
        wC  = st.slider("Peso afinidade",          0.0, 1.0, 0.10, 0.05)
        wCH = st.slider("🌀 Peso Teoria do Caos",  0.0, 1.0, 0.10, 0.05)

    pesos = (w50, w20, wA, wR, wC, wCH)
    soma_pesos = sum(pesos)
    if abs(soma_pesos - 1.0) > 0.01:
        st.warning(f"⚠️ Soma dos pesos: {soma_pesos:.2f}. Recomenda-se somar 1.0.")
    else:
        st.success(f"✅ Soma dos pesos: {soma_pesos:.2f}")

    st.info(
        "**🌀 Peso Teoria do Caos:** incorpora Expoente de Hurst, Lyapunov e Entropia de "
        "Permutação por dezena. Dezenas em regime *Persistente* com baixa entropia recebem "
        "ChaosScore mais alto, aumentando sua probabilidade de seleção."
    )

freq50, freq20, atraso, repeticoes, afinidade, caos, score, prob = calcular_estatisticas(dezenas_df, pesos)

# ----------------------------------------------------------------
# Aba: Análise de Dezenas
# ----------------------------------------------------------------
with tab_dezenas:
    st.subheader("📊 Ranking de dezenas por score")
    ranking = pd.DataFrame({
        "Dezena":         [int(n) for n in TODAS_DEZENAS],
        "Score":          [score[int(n)] for n in TODAS_DEZENAS],
        "Freq50":         [freq50[int(n)] for n in TODAS_DEZENAS],
        "Freq20":         [freq20[int(n)] for n in TODAS_DEZENAS],
        "Atraso":         [atraso[int(n)] for n in TODAS_DEZENAS],
        "🌀 ChaosScore":  [caos[int(n)]["chaos_score"] for n in TODAS_DEZENAS],
        "Regime":         [caos[int(n)]["regime"] for n in TODAS_DEZENAS],
    }).sort_values("Score", ascending=False)

    st.dataframe(ranking, use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = cm.RdYlGn([caos[int(n)]["chaos_score"] for n in ranking["Dezena"]])
    ax.bar(ranking["Dezena"], ranking["Score"], color=colors)
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Score Total")
    ax.set_title("Score por Dezena  (cor = ChaosScore: 🔴 baixo → 🟢 alto)")
    st.pyplot(fig)

# ----------------------------------------------------------------
# Aba: 🌀 Teoria do Caos
# ----------------------------------------------------------------
with tab_caos:
    st.subheader("🌀 Análise de Complexidade — Teoria do Caos por Dezena")
    st.markdown("""
| Métrica | O que mede | Interpretação |
|---|---|---|
| **Hurst (H)** | Persistência da série | H>0.55 persistente · H<0.45 reversão · H≈0.5 aleatório |
| **Lyapunov (λ)** | Sensibilidade às condições iniciais | λ>0 caótico · λ<0 estável |
| **Entropia (E)** | Complexidade local | E→1 caótico · E→0 regular |
| **Regime** | Classificação automática | Persistente / Anti-persistente / Estocástico / Caótico |
| **Oportunidade** | Score de aproveitamento | Quanto maior, mais favorável para previsão |
    """)

    caos_df = pd.DataFrame([
        {
            "Dezena":       n,
            "Hurst (H)":    caos[n]["hurst"],
            "Lyapunov (λ)": caos[n]["lyapunov"],
            "Entropia (E)": caos[n]["entropia"],
            "Regime":       caos[n]["regime"],
            "Oportunidade": caos[n]["oportunidade"],
            "ChaosScore":   caos[n]["chaos_score"],
        }
        for n in sorted(caos.keys())
    ])

    col_f, col_s = st.columns(2)
    with col_f:
        regimes_disp = caos_df["Regime"].unique().tolist()
        regime_sel   = st.multiselect("Filtrar por regime", regimes_disp, default=regimes_disp)
    with col_s:
        col_ordem = st.selectbox("Ordenar por",
            ["ChaosScore", "Hurst (H)", "Lyapunov (λ)", "Entropia (E)", "Oportunidade"])

    st.dataframe(
        caos_df[caos_df["Regime"].isin(regime_sel)].sort_values(col_ordem, ascending=False),
        use_container_width=True
    )

    # Mapa de Regime
    st.markdown("#### 🔬 Mapa de Regime: Hurst × Entropia de Permutação")
    st.caption("Cada ponto é uma dezena. Posição no mapa revela seu comportamento dinâmico.")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sc = ax2.scatter(
        caos_df["Hurst (H)"], caos_df["Entropia (E)"],
        c=[caos[n]["chaos_score"] for n in sorted(caos.keys())],
        cmap="RdYlGn", s=110, edgecolors="black", linewidths=0.5
    )
    for _, row in caos_df.iterrows():
        ax2.annotate(str(int(row["Dezena"])), (row["Hurst (H)"], row["Entropia (E)"]),
                     fontsize=7, ha="center", va="bottom")
    plt.colorbar(sc, ax=ax2, label="ChaosScore")
    ax2.axvline(0.5,  color="gray",   linestyle="--", linewidth=0.8, label="H=0.5 (aleatório)")
    ax2.axhline(0.75, color="orange", linestyle="--", linewidth=0.8, label="E=0.75 (limiar caótico)")
    ax2.set_xlabel("Expoente de Hurst (H)")
    ax2.set_ylabel("Entropia de Permutação (E)")
    ax2.set_title("Mapa de Regime Caótico das Dezenas")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

    # Hurst Deslizante
    st.markdown("#### 📉 Evolução Temporal do Hurst (janela deslizante)")
    st.caption("Como o regime de persistência de cada dezena evolui ao longo dos concursos.")
    top_dezenas = caos_df.sort_values("ChaosScore", ascending=False).head(8)["Dezena"].tolist()
    janela  = st.slider("Janela de análise (concursos)", 30, 200, 80, step=10)
    n_total = len(dezenas_df)
    passo   = max(1, janela // 4)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    for dez in top_dezenas:
        serie = _serie_temporal_dezena(dezenas_df, int(dez))
        ht    = [calcular_hurst(serie[i - janela:i]) for i in range(janela, n_total, passo)]
        ax3.plot(list(range(janela, n_total, passo)), ht, marker="o", markersize=2, label=f"Dezena {dez}")
    ax3.axhline(0.5,  color="gray",  linestyle="--", linewidth=0.8, label="H=0.5")
    ax3.axhline(0.55, color="green", linestyle=":",  linewidth=0.8, label="H=0.55 (persistência)")
    ax3.axhline(0.45, color="red",   linestyle=":",  linewidth=0.8, label="H=0.45 (anti-persist.)")
    ax3.set_xlabel("Concurso (índice)")
    ax3.set_ylabel("Expoente de Hurst")
    ax3.set_title("Hurst Deslizante — Top 8 Dezenas por ChaosScore")
    ax3.legend(fontsize=7, ncol=2)
    st.pyplot(fig3)

    # Alerta de oportunidade
    st.markdown("#### 🎯 Dezenas em Maior Oportunidade Agora")
    st.dataframe(
        caos_df.sort_values("Oportunidade", ascending=False).head(10)
               [["Dezena","Regime","Hurst (H)","Entropia (E)","Oportunidade","ChaosScore"]],
        use_container_width=True
    )

# ----------------------------------------------------------------
# Aba: Jogos Otimizados
# ----------------------------------------------------------------
with tab_jogos:
    st.subheader("🎲 Geração de Jogos Otimizados")
    num_sim    = st.slider("Número de simulações Monte Carlo", 5000, 100000, 20000)
    qtd_jogos  = st.number_input("Quantos jogos deseja gerar?", 1, 30, 10)
    min_pares  = st.number_input("Mínimo de pares", 0, 6, 2)
    max_pares  = st.number_input("Máximo de pares", 0, 6, 4)
    soma_min   = st.number_input("Soma mínima", 60, 300, 120)
    soma_max   = st.number_input("Soma máxima", 60, 300, 210)
    evitar_rep = st.checkbox("Evitar repetir dezenas entre jogos", True)
    filtros    = (min_pares, max_pares, soma_min, soma_max)

    if st.button("Gerar jogos"):
        jogos = gerar_jogos(prob, dezenas_df, qtd_jogos, num_sim, filtros, evitar_rep)
        if not jogos:
            st.warning("Nenhum jogo passou pelos filtros. Tente afrouxar os critérios.")
        else:
            jogos_df = pd.DataFrame(jogos, columns=["D1","D2","D3","D4","D5","D6"])
            jogos_df["🌀 ChaosScore Médio"] = [
                round(np.mean([caos[n]["chaos_score"] for n in jogo]), 4) for jogo in jogos
            ]
            jogos_df["Regime Dominante"] = [
                pd.Series([caos[n]["regime"] for n in jogo]).mode()[0] for jogo in jogos
            ]
            st.dataframe(jogos_df, use_container_width=True)

            fig4, ax4 = plt.subplots(figsize=(6, 3))
            ax4.hist([sum(j) for j in jogos], bins=10, edgecolor="black", color="steelblue")
            ax4.set_xlabel("Soma dos jogos")
            ax4.set_ylabel("Frequência")
            st.pyplot(fig4)

            from io import BytesIO
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine="xlsxwriter")
            jogos_df.to_excel(writer, index=False, sheet_name="Jogos")
            writer.close()
            st.download_button("📥 Baixar jogos em Excel", output.getvalue(),
                               "jogos_mega_otimizados.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------------------------------------------
# Aba: Pacotes de Perfis
# ----------------------------------------------------------------
with tab_pacotes:
    st.subheader("🎯 Pacotes de jogos por perfil")
    num_sim_pac    = st.slider("Simulações (pacotes)", 5000, 50000, 15000)
    qtd_por_perfil = st.number_input("Jogos por perfil", 1, 20, 5)
    perfis = {
        "Conservador": (2, 4, 120, 200),
        "Equilibrado": (2, 4, 130, 220),
        "Agressivo":   (1, 5, 150, 260),
    }
    if st.button("Gerar pacotes"):
        for nome, filtros_p in perfis.items():
            jogos = gerar_jogos(prob, dezenas_df, qtd_por_perfil, num_sim_pac, filtros_p, evitar_repeticao=True)
            st.markdown(f"**Perfil {nome}:**")
            if jogos:
                dfp = pd.DataFrame(jogos, columns=["D1","D2","D3","D4","D5","D6"])
                dfp["🌀 ChaosScore Médio"] = [
                    round(np.mean([caos[n]["chaos_score"] for n in jogo]), 4) for jogo in jogos
                ]
                st.dataframe(dfp, use_container_width=True)
            else:
                st.warning(f"Nenhum jogo encontrado para o perfil {nome}.")

# ----------------------------------------------------------------
# Aba: Backtest & Métricas
# ----------------------------------------------------------------
with tab_backtest:
    st.subheader("📈 Backtest do modelo")
    max_idx            = len(dezenas_df) - 1
    inicio             = st.slider("Concurso inicial (índice)", 0, max_idx-10, max_idx-100)
    fim                = st.slider("Concurso final (índice)",   inicio+10, max_idx, max_idx)
    num_sim_bt         = st.slider("Simulações por concurso", 1000, 20000, 5000)
    jogos_por_concurso = st.number_input("Jogos por concurso no backtest", 1, 20, 5)
    min_pares_bt       = st.number_input("Mínimo de pares (BT)", 0, 6, 2)
    max_pares_bt       = st.number_input("Máximo de pares (BT)", 0, 6, 4)
    soma_min_bt        = st.number_input("Soma mínima (BT)", 60, 300, 120)
    soma_max_bt        = st.number_input("Soma máxima (BT)", 60, 300, 210)
    filtros_bt         = (min_pares_bt, max_pares_bt, soma_min_bt, soma_max_bt)

    if st.button("Rodar backtest"):
        bt_df = backtest(dezenas_df, pesos, num_sim_bt, jogos_por_concurso, inicio, fim, filtros_bt)
        if bt_df.empty:
            st.warning("Nenhum resultado de backtest gerado.")
        else:
            st.dataframe(bt_df, use_container_width=True)
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Média de acertos por jogo", f"{bt_df['acertos'].mean():.2f}")
            col_m2.metric("Máximo de acertos em um jogo", str(bt_df["acertos"].max()))
            dist = bt_df["acertos"].value_counts().sort_index()
            fig5, ax5 = plt.subplots(figsize=(6, 3))
            ax5.bar(dist.index, dist.values, color="steelblue", edgecolor="black")
            ax5.set_xlabel("Número de acertos")
            ax5.set_ylabel("Quantidade de jogos")
            ax5.set_title("Distribuição de Acertos no Backtest")
            st.pyplot(fig5)