"""Scénarios explicites de préparation de la matrice client-produit.

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
    """Scénario principal : partie initiale unique par id ; index = id."""
    verifier_produits(df, produits)
    initial, _, _ = decouper_a_la_premiere_rupture(df)
    if initial['id'].duplicated().any():
        raise ValueError('Le segment initial contient des ID répétés.')
    return initial.set_index('id')[produits].astype('int8')


def base_agregee(df: pd.DataFrame, produits: list[str]) -> pd.DataFrame:
    """Scénario alternatif : présence d'un produit si au moins une ligne vaut 1."""
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