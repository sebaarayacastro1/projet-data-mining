"""Calculs de la matrice et des indicateurs du cours « Les associations ».

Une ligne = un identifiant client ; une colonne = un produit binaire.
Les règles considérées ici relient deux produits distincts (A -> B).
"""

import numpy as np
import pandas as pd


def verifier_matrice_clients(base: pd.DataFrame) -> None:
    """Vérifie que la matrice d'analyse contient uniquement des produits binaires."""
    if base.empty or base.shape[1] < 2:
        raise ValueError('La matrice doit contenir des clients et au moins deux produits.')
    if not base.index.is_unique or base.index.hasnans:
        raise ValueError('Une seule ligne par identifiant client est attendue.')
    if base.columns.has_duplicates or any(c in base.columns for c in ('id', 'SEXE')):
        raise ValueError('Les colonnes doivent être des produits distincts, sans id ni SEXE.')
    if base.isna().any().any() or not base.isin([0, 1, False, True]).all().all():
        raise ValueError('Chaque produit doit prendre la valeur 0 ou 1, sans valeur manquante.')


def matrice_association(base: pd.DataFrame) -> pd.DataFrame:
    """Effectifs des co-détentions ; la diagonale donne N(A)."""
    verifier_matrice_clients(base)
    indicateurs = base.astype(np.int64)
    effectifs = indicateurs.T.dot(indicateurs)
    effectifs.index.name = 'produit_1'
    effectifs.columns.name = 'produit_2'
    return effectifs


def indicateurs_association(base: pd.DataFrame) -> pd.DataFrame:
    """Support, confiance, confiance attendue et lift pour chaque A -> B.

    Règles avec A différent de B ; celles dont N(A) ou N(B) vaut zéro
    reçoivent une valeur manquante pour les ratios non définis.
    """
    effectifs = matrice_association(base)
    total = len(base)
    produits = list(base.columns)
    rows = []
    for a in produits:
        na = int(effectifs.loc[a, a])
        for b in produits:
            if a == b:
                continue
            nb = int(effectifs.loc[b, b])
            nab = int(effectifs.loc[a, b])
            support = nab / total
            confiance = nab / na if na else np.nan
            attendue = nb / total
            lift = confiance / attendue if na and nb else np.nan
            rows.append({
                'produit_a': a, 'produit_b': b,
                'nb_clients_a': na, 'nb_clients_b': nb, 'nb_clients_a_et_b': nab,
                'support': support, 'confiance': confiance,
                'confiance_attendue': attendue, 'lift': lift,
            })
    return pd.DataFrame.from_records(rows)


def comparer_indicateurs(regles_initiales: pd.DataFrame,
                        regles_agregees: pd.DataFrame) -> pd.DataFrame:
    """Comparer les mêmes associations A -> B entre les deux scénarios."""
    mesures = ['support', 'confiance', 'confiance_attendue', 'lift']
    cles = ['produit_a', 'produit_b']
    a = regles_initiales[cles + mesures].set_index(cles)
    b = regles_agregees[cles + mesures].set_index(cles)
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError('Les deux scénarios doivent contenir les mêmes règles uniques.')
    res = a.join(b, lsuffix='_initial', rsuffix='_agregation', how='inner')
    for mesure in mesures:
        res[f'ecart_{mesure}'] = (res[f'{mesure}_agregation']
                                  - res[f'{mesure}_initial'])
    return res.reset_index()
