"""Tests de la consolidation des profils client-produit."""
import unittest
import pandas as pd

from src.preparation import (
    base_agregee, diagnostic_identifiants, verifier_base_agregee,
)


class TestPreparation(unittest.TestCase):
    def setUp(self):
        # ID 10 : deux profils complémentaires ; ID 20 et 30 : même profil.
        self.df = pd.DataFrame({
            'id': [10, 10, 20, 30],
            'SEXE': ['M', 'M', 'F', 'F'],
            'DAV': [1, 1, 1, 1],
            'ASSMH': [0, 1, 1, 1],
        })
        self.produits = ['DAV', 'ASSMH']

    def test_agregation_une_ligne_par_identifiant(self):
        base = base_agregee(self.df, self.produits)
        self.assertEqual(len(base), 3)
        self.assertEqual(base.loc[10, 'ASSMH'], 1)
        self.assertEqual(base.loc[10, 'DAV'], 1)
        verifier_base_agregee(self.df, self.produits, base)

    def test_ids_differents_et_profils_identiques_conserves(self):
        base = base_agregee(self.df, self.produits)
        self.assertIn(20, base.index)
        self.assertIn(30, base.index)
        self.assertTrue(base.loc[20].equals(base.loc[30]))

    def test_diagnostic(self):
        bilan = diagnostic_identifiants(self.df, self.produits)
        resultats = dict(zip(bilan['indicateur'], bilan['valeur']))
        self.assertEqual(resultats['identifiants_repetes'], 1)
        self.assertEqual(resultats['lignes_excedentaires_par_id'], 1)
        self.assertEqual(resultats['ids_repetes_profils_produits_differents'], 1)

    def test_refus_dune_valeur_non_binaire(self):
        invalide = self.df.copy()
        invalide.loc[0, 'DAV'] = 2
        with self.assertRaises(ValueError):
            base_agregee(invalide, self.produits)

    def test_refus_matrice_incomplete(self):
        base = base_agregee(self.df, self.produits).drop(index=30)
        with self.assertRaises(ValueError):
            verifier_base_agregee(self.df, self.produits, base)


if __name__ == '__main__':
    unittest.main()
