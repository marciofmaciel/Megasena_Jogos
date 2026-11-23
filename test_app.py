import io
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


class AplicativoTest(unittest.TestCase):
    def test_geracao_nas_duas_versoes(self):
        dados = pd.DataFrame([
            [1, 'data', 2, 4, 6, 8, 10, 12],
            [2, 'data', 3, 5, 7, 9, 11, 13],
        ], columns=['Concurso', 'Data', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6'])
        arquivo = io.BytesIO()
        dados.to_excel(arquivo, index=False)
        conteudo = arquivo.getvalue()

        def uploader(*args, **kwargs):
            return None if kwargs.get('key') == 'lf_uploader' else io.BytesIO(conteudo)

        for arquivo_app in ('app.py', 'app_V1.py'):
            with self.subTest(app=arquivo_app), patch('streamlit.file_uploader', side_effect=uploader):
                app = AppTest.from_file(arquivo_app, default_timeout=120).run()
                self.assertFalse(app.exception, str(app.exception))
                next(b for b in app.button if b.label == 'Gerar jogos').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertTrue(any('Senas distintas: 10' in c.value for c in app.caption))
                next(b for b in app.button if b.label == 'Gerar pacotes').click().run()
                self.assertFalse(app.exception, str(app.exception))
                self.assertEqual(sum('Senas distintas: 5' in c.value for c in app.caption), 3)
                next(b for b in app.button if b.label == 'Rodar backtest').click().run()
                self.assertFalse(app.exception, str(app.exception))
                if arquivo_app == 'app.py':
                    next(b for b in app.button if b.label == '🎯 Gerar jogos do universo').click().run()
                    self.assertFalse(app.exception, str(app.exception))
                    self.assertTrue(any('Senas distintas: 3' in c.value for c in app.caption))


if __name__ == '__main__':
    unittest.main()
