"""Geracao de jogos da Mega-Sena por cobertura combinatoria, sem previsao."""

from itertools import combinations
from math import comb
import random


UNIVERSO = tuple(range(1, 61))
TOTAL_RESULTADOS = comb(60, 6)


def historico_de_jogos(linhas):
    """Valida cada concurso; nao ignora linhas invalidas silenciosamente."""
    historico = set()
    for indice, linha in enumerate(linhas, 1):
        valores = tuple(linha)
        try:
            numeros = tuple(int(x) for x in valores)
            validos = all(float(x) == n and 1 <= n <= 60 for x, n in zip(valores, numeros))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Concurso na linha {indice}: dezenas invalidas.") from exc
        if len(numeros) != 6 or not validos or len(set(numeros)) != 6:
            raise ValueError(f"Concurso na linha {indice}: sao necessarias seis dezenas distintas de 1 a 60.")
        historico.add(tuple(sorted(numeros)))
    return historico


def tem_sequencia(jogo):
    """Considera sequencia uma serie de tres ou mais dezenas consecutivas."""
    return any(b == a + 1 and c == b + 1 for a, b, c in zip(jogo, jogo[1:], jogo[2:]))


def popularidade_estimada(jogo):
    """Heuristica, nao uma medicao das apostas de outros jogadores."""
    return (sum(n <= 31 for n in jogo), sum(n % 5 == 0 for n in jogo))


def resultados_cobertos(jogo):
    """Resultados de seis dezenas que dariam quina ou sena para este jogo."""
    fora = set(UNIVERSO).difference(jogo)
    yield jogo
    for quina in combinations(jogo, 5):
        for numero in fora:
            yield tuple(sorted((*quina, numero)))


def metricas_cobertura(jogos):
    cobertos = set()
    quinas = set()
    for jogo in jogos:
        cobertos.update(resultados_cobertos(jogo))
        quinas.update(combinations(jogo, 5))
    senas = len(set(jogos))
    return {
        "senas": senas,
        "quinas_distintas": len(quinas),
        "resultados_quina_ou_sena": len(cobertos),
        "resultados_quina_exata": len(cobertos) - senas,
        "total_resultados": TOTAL_RESULTADOS,
    }


def gerar_jogos_cobertura(linhas, quantidade, candidatos, filtros=(0, 6, 21, 345),
                           evitar_repeticao=False, universo=UNIVERSO, seed=None):
    """Busca gulosa: maximiza ganho de resultados com >=5 acertos na amostra.

    Empates favorecem jogos com menos dezenas de calendario e multiplos de 5.
    O otimo global nao e garantido; candidatos sao amostrados uniformemente.
    """
    if quantidade < 1 or candidatos < 1:
        raise ValueError("A quantidade de jogos e de candidatos deve ser positiva.")
    minimo, maximo, soma_min, soma_max = filtros
    if not (0 <= minimo <= maximo <= 6 and 21 <= soma_min <= soma_max <= 345):
        raise ValueError("Verifique os limites de pares e de soma.")
    universo = tuple(sorted(universo))
    if len(universo) < 6 or len(set(universo)) != len(universo) or any(n not in UNIVERSO for n in universo):
        raise ValueError("O universo deve conter ao menos seis dezenas distintas de 1 a 60.")
    historico = historico_de_jogos(linhas)
    rng = random.Random(seed)
    amostra = set()
    for _ in range(candidatos):
        jogo = tuple(sorted(rng.sample(universo, 6)))
        pares = sum(n % 2 == 0 for n in jogo)
        if (jogo not in historico and not tem_sequencia(jogo)
                and minimo <= pares <= maximo and soma_min <= sum(jogo) <= soma_max):
            amostra.add(jogo)

    # Cada jogo cobre 325 resultados. Intersecoes de ate tres dezenas nao
    # compartilham nenhum resultado com cinco ou mais acertos.
    ordenados = sorted(amostra)
    rng.shuffle(ordenados)
    ordenados.sort(key=popularidade_estimada)
    jogos = []
    cobertos = set()
    usadas = set()
    while len(jogos) < quantidade:
        melhor = None
        melhor_ganho = -1
        for jogo in ordenados:
            if jogo in jogos or (evitar_repeticao and not usadas.isdisjoint(jogo)):
                continue
            if all(len(set(jogo).intersection(outro)) <= 3 for outro in jogos):
                ganho = 325
            else:
                ganho = sum(resultado not in cobertos for resultado in resultados_cobertos(jogo))
            if ganho > melhor_ganho:
                melhor, melhor_ganho = jogo, ganho
                if ganho == 325:
                    break
        if melhor is None:
            break
        jogos.append(melhor)
        usadas.update(melhor)
        cobertos.update(resultados_cobertos(melhor))
    return jogos
