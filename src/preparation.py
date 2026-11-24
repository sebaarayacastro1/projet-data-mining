"""Préparation et contrôle de matrices client-produit.

Aucune suppression définitive ni modification du fichier d'origine.
"""
import pandas as pd


def verifier_produits(df: pd.DataFrame, produits: list[str]) -> None:
    """Vérifier que les produits sont binaires et ne contiennent pas de valeurs absentes."""
    if not produits:
        raise ValueError('Aucune variable produit disponible.')
    if df['id'].isna().any():
        raise ValueError('Identifiants manquants : examiner les données avant agrégation.')
    for colonne in produits:
        if colonne not in df.columns or not df[colonne].isin([0, 1]).all():
            raise ValueError(f'Variable non binaire ou absente : {colonne}')


def decouper_a_la_premiere_rupture(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Découpage exploratoire fondé sur la première baisse d'ID (pas une preuve d'erreur)."""
    ruptures = df.index[df['id'].diff().lt(0)].tolist()
    if not ruptures:
        raise ValueError('Aucune rupture décroissante des identifiants.')
    position = df.index.get_loc(ruptures[0])
    return df.iloc[:position].copy(), df.iloc[position:].copy(), position


def base_initiale(df: pd.DataFrame, produits: list[str]) -> pd.DataFrame:
    """Scénario de référence : partie initiale unique par id ; index = id."""
    verifier_produits(df, produits)
    initial, _, _ = decouper_a_la_premiere_rupture(df)
    if initial['id'].duplicated().any():
        raise ValueError('Le segment initial contient des ID répétés.')
    return initial.set_index('id')[produits].astype('int8')


def base_agregee(df: pd.DataFrame, produits: list[str]) -> pd.DataFrame:
    """Scénario principal : produit présent si au moins une ligne du même ID vaut 1.

    Hypothèse de travail : les lignes d'un même ID sont consolidables.
    Cette hypothèse n'est pas confirmée par le fichier source.
    """
    verifier_produits(df, produits)
    return df.groupby('id', sort=True)[produits].max().astype('int8')


def frequences_produits(base: pd.DataFrame) -> pd.DataFrame:
    """Part de clients détenant chaque produit, dans la base choisie."""
    return (base.mean().mul(100).sort_values(ascending=False)
            .rename('pourcentage_clients').to_frame())


def comparer_frequences(base_a: pd.DataFrame, base_b: pd.DataFrame) -> pd.DataFrame:
    """Comparaison descriptive des deux scénarios, en points de pourcentage."""
    if list(base_a.columns) != list(base_b.columns):
        raise ValueError('Les scénarios doivent contenir les mêmes produits.')
    result = pd.DataFrame({
        'segment_initial_pct': base_a.mean().mul(100),
        'agregation_pct': base_b.mean().mul(100),
    })
    result['ecart_points'] = result['agregation_pct'] - result['segment_initial_pct']
    return result.sort_values('ecart_points', key=lambda s: s.abs(), ascending=False)

def base_identifiants_distincts(df, produits):
    """
    Conserve la première observation de chaque identifiant.

    Hypothèse : chaque identifiant représente un client.
    """
    return (
        df.drop_duplicates(subset="id", keep="first")
        .set_index("id")[produits]
        .copy()
    )

def diagnostic_identifiants(df: pd.DataFrame, produits: list[str]) -> pd.DataFrame:
    """Résumé des identifiants répétés et de leurs profils de produits."""
    verifier_produits(df, produits)
    tailles = df.groupby('id').size()
    repetes = tailles[tailles > 1].index
    # Chaque ligne représente le profil binaire complet d'un identifiant.
    configurations = (
        df.loc[df['id'].isin(repetes), ['id', *produits]]
        .drop_duplicates()
        .groupby('id')
        .size()
    )
    lignes = {
        'observations_initiales': len(df),
        'identifiants_distincts': df['id'].nunique(),
        'identifiants_repetes': len(repetes),
        'lignes_excedentaires_par_id': len(df) - df['id'].nunique(),
        'ids_repetes_profils_produits_differents': int((configurations > 1).sum()),
    }
    return pd.DataFrame(
        {'indicateur': list(lignes.keys()), 'valeur': list(lignes.values())}
    )


def verifier_base_agregee(
    df: pd.DataFrame, produits: list[str], base: pd.DataFrame
) -> None:
    """Vérifier la cohérence technique de la matrice agrégée."""
    verifier_produits(df, produits)
    if base.index.has_duplicates:
        raise ValueError('La matrice agrégée contient des identifiants répétés.')
    if len(base) != df['id'].nunique():
        raise ValueError('Le nombre de lignes ne correspond pas aux ID distincts.')
    if list(base.columns) != list(produits):
        raise ValueError('Les colonnes produits ne correspondent pas au fichier source.')
    if not base.isin([0, 1]).all().all():
        raise ValueError('La matrice agrégée contient des valeurs non binaires.')
    if set(base.index) != set(df['id']):
        raise ValueError('Certains identifiants ont été perdus ou ajoutés.')
