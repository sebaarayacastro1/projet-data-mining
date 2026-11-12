"""Chargement du CSV original sans modification."""
from pathlib import Path
import pandas as pd

def trouver_donnees(racine: Path) -> Path:
    chemin = racine / 'data' / 'données.csv'
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier absent : {chemin}. Placer le CSV original dans data/.")
    return chemin

def charger_donnees(racine: Path) -> pd.DataFrame:
    return pd.read_csv(trouver_donnees(racine), sep=';')

def colonnes_produits(df: pd.DataFrame) -> list[str]:
    if not {'id', 'SEXE'}.issubset(df.columns):
        raise ValueError("Les colonnes 'id' et 'SEXE' sont attendues.")
    return [c for c in df.columns if c not in ('id', 'SEXE')]
