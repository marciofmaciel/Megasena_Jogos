import itertools
import unittest

from cobertura import (
    gerar_jogos_cobertura, historico_de_jogos, metricas_cobertura,
    popularidade_estimada, resultados_cobertos, tem_sequencia,
)


class CoberturaTest(unittest.TestCase):
    def test_cobertura_de_um_jogo(self):
        jogo = (1, 3, 5, 7, 9, 11)
        resultados = set(resultados_cobertos(jogo))
        self.assertEqual(len(resultados), 325)
        self.assertTrue(all(len(set(jogo) & set(resultado)) >= 5 for resultado in resultados))
        metricas = metricas_cobertura([jogo])
        self.assertEqual(metricas['senas'], 1)
        self.assertEqual(metricas['quinas_distintas'], 6)
        self.assertEqual(metricas['resultados_quina_exata'], 324)
        self.assertEqual(metricas['total_resultados'], 50063860)

    def test_sobreposicao_e_quina_exata(self):
        a = (1, 3, 5, 7, 9, 11)
        for b, esperado in [((1, 3, 5, 7, 9, 13), 590),
                            ((1, 3, 5, 7, 13, 15), 646),
                            ((1, 3, 5, 13, 15, 17), 650)]:
            with self.subTest(b=b):
                metricas = metricas_cobertura([a, b])
                self.assertEqual(metricas['resultados_quina_ou_sena'], esperado)
                self.assertEqual(metricas['resultados_quina_exata'], esperado - 2)

    def test_historico_validacao_e_exclusao(self):
        jogo = (2, 4, 6, 8, 10, 12)
        self.assertEqual(historico_de_jogos([reversed(jogo)]), {jogo})
        self.assertEqual(gerar_jogos_cobertura([jogo], 1, 10, universo=jogo, seed=3), [])
        for invalido in [(1, 2, 3, 4, 5, 5), (1, 2, 3, 4, 5, 61),
                         (1, 2, 3, 4, 5, 6.5), (1, 2, 3, 4, 5, None)]:
            with self.subTest(invalido=invalido), self.assertRaises(ValueError):
                historico_de_jogos([invalido])

    def test_geracao_e_limite_de_cobertura(self):
        historico = [(1, 2, 3, 4, 5, 6)]
        jogos = gerar_jogos_cobertura(historico, 30, 20000, seed=42)
        self.assertEqual(len(jogos), 30)
        self.assertEqual(len(set(jogos)), 30)
        self.assertTrue(all(jogo not in historico and not tem_sequencia(jogo) for jogo in jogos))
        self.assertTrue(all(len(jogo) == len(set(jogo)) == 6 and 1 <= min(jogo) <= max(jogo) <= 60 for jogo in jogos))
        self.assertEqual(metricas_cobertura(jogos)['resultados_quina_ou_sena'], 30 * 325)
        self.assertEqual(jogos, gerar_jogos_cobertura(historico, 30, 20000, seed=42))

    def test_ganho_guloso_em_universo_pequeno(self):
        universo = (2, 4, 6, 8, 10, 12, 14, 16)
        candidatos = list(itertools.combinations(universo, 6))
        jogos = gerar_jogos_cobertura([], 4, 2000, universo=universo, seed=1)
        cobertos = set()
        for jogo in jogos:
            ganho = len(set(resultados_cobertos(jogo)) - cobertos)
            self.assertEqual(ganho, max(len(set(resultados_cobertos(c)) - cobertos)
                                        for c in candidatos if c not in jogos[:jogos.index(jogo)]))
            cobertos.update(resultados_cobertos(jogo))

    def test_filtros_repeticao_e_resultado_parcial(self):
        jogos = gerar_jogos_cobertura([], 12, 10000, (2, 4, 120, 210), True, seed=7)
        self.assertLessEqual(len(jogos), 10)
        self.assertEqual(len(set(itertools.chain.from_iterable(jogos))), len(jogos) * 6)
        self.assertTrue(all(2 <= sum(n % 2 == 0 for n in jogo) <= 4 and 120 <= sum(jogo) <= 210 for jogo in jogos))
        with self.assertRaises(ValueError):
            gerar_jogos_cobertura([], 1, 10, (4, 2, 21, 345))

    def test_sequencias_e_popularidade(self):
        self.assertTrue(tem_sequencia((1, 2, 3, 10, 12, 14)))
        self.assertFalse(tem_sequencia((1, 2, 4, 6, 8, 10)))
        self.assertLess(popularidade_estimada((32, 33, 37, 41, 47, 53)),
                        popularidade_estimada((1, 2, 7, 11, 17, 23)))


if __name__ == '__main__':
    unittest.main()
