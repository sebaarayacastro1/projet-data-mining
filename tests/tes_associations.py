"""Vérification sur les 10 clients de l'exemple du cours (page 7)."""
import unittest
import pandas as pd

from src.associations import (
    matrice_association, indicateurs_association, comparer_indicateurs
)


class TestAssociations(unittest.TestCase):
    def setUp(self):
        # Exemple : N(A)=4, N(B)=5, N(C)=7, N(A,B)=3, N(A,C)=1, N(B,C)=3
        paniers = {
            1: ('A', 'B'), 2: ('B', 'C'), 3: ('A', 'B'),
            4: ('C',), 5: ('B', 'C'), 6: ('A', 'B', 'C'),
            7: ('C',), 8: ('C',), 9: ('C',), 10: ('A',),
        }
        self.base = pd.DataFrame(
            [{p: int(p in paniers[client]) for p in ('A', 'B', 'C')}
             for client in paniers], index=list(paniers),
        )
        self.base.index.name = 'id'

    def test_matrice_du_cours(self):
        m = matrice_association(self.base)
        self.assertEqual(m.loc['A', 'A'], 4)
        self.assertEqual(m.loc['B', 'B'], 5)
        self.assertEqual(m.loc['C', 'C'], 7)
        self.assertEqual(m.loc['A', 'B'], 3)
        self.assertEqual(m.loc['A', 'C'], 1)
        self.assertEqual(m.loc['B', 'C'], 3)
        self.assertTrue((m == m.T.rename_axis(index='produit_1', columns='produit_2')).all().all())

    def test_mesures_du_cours(self):
        r = indicateurs_association(self.base).set_index(['produit_a', 'produit_b'])
        ab = r.loc[('A', 'B')]
        self.assertAlmostEqual(ab['support'], .30)
        self.assertAlmostEqual(ab['confiance'], .75)
        self.assertAlmostEqual(ab['confiance_attendue'], .50)
        self.assertAlmostEqual(ab['lift'], 1.5)
        ba = r.loc[('B', 'A')]
        self.assertAlmostEqual(ba['confiance'], .60)
        self.assertAlmostEqual(ba['lift'], 1.5)
        self.assertEqual(len(r), 6)

    def test_comparaison_identique(self):
        r = indicateurs_association(self.base)
        c = comparer_indicateurs(r, r)
        self.assertTrue((c['ecart_support'] == 0).all())
        self.assertTrue((c['ecart_lift'] == 0).all())

    def test_rejeter_identifiants_doubles(self):
        base = self.base.copy()
        base.index = [1] * len(base)
        with self.assertRaises(ValueError):
            matrice_association(base)


if __name__ == '__main__':
    unittest.main()
