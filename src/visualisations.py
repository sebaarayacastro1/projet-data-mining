"""Figures descriptives pour l'analyse des associations de produits.

Les fonctions affichent les figures et peuvent les enregistrer dans reports/figures/.
Aucun seuil commercial définitif n'est imposé ici.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _enregistrer(fig, chemin):
    if chemin is not None:
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=180, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def graphique_frequences(base: pd.DataFrame, chemin=None):
    """Part de clients détenant chaque produit, en pourcentage."""
    frequences = (base.mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, len(frequences) * .28)))
    ax.barh(frequences.index, frequences.values)
    ax.set_xlabel('Clients détenant le produit (%)')
    ax.set_title('Fréquence de détention des produits — base agrégée')
    ax.grid(axis='x', alpha=.2)
    _enregistrer(fig, chemin)


def graphique_matrice_cooccurrences(effectifs: pd.DataFrame, chemin=None):
    """Nombre de clients détenant conjointement deux produits."""
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(effectifs.to_numpy(), aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(effectifs.columns)), labels=effectifs.columns, rotation=90)
    ax.set_yticks(range(len(effectifs.index)), labels=effectifs.index)
    ax.set_title('Co-détention des produits (nombre de clients)')
    fig.colorbar(image, ax=ax, label='Nombre de clients')
    _enregistrer(fig, chemin)


def graphique_lift_support(regles: pd.DataFrame, chemin=None):
    """Relation support-lift pour les associations au lift défini."""
    valides = regles.loc[np.isfinite(regles['lift'])].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(valides['support'] * 100, valides['lift'], s=20, alpha=.5)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Support (%)')
    ax.set_ylabel('Lift')
    ax.set_title('Support et lift des associations orientées')
    ax.grid(alpha=.2)
    _enregistrer(fig, chemin)


def graphique_comparaison_support(comparaison: pd.DataFrame, chemin=None, n=12):
    """Associations présentant les plus grands écarts absolus de support."""
    selection = comparaison.copy()
    selection['ecart_abs'] = selection['ecart_support'].abs() * 100
    selection = selection.nlargest(n, 'ecart_abs').sort_values('ecart_abs')
    etiquettes = selection['produit_a'] + ' → ' + selection['produit_b']
    fig, ax = plt.subplots(figsize=(11, max(4, len(selection) * .4)))
    ax.barh(etiquettes, selection['ecart_support'] * 100)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Écart de support : base agrégée − segment initial (points de %)')
    ax.set_title('Associations les plus sensibles au choix de la base')
    ax.grid(axis='x', alpha=.2)
    _enregistrer(fig, chemin)



def _carte_annotee(matrice: pd.DataFrame, titre: str, chemin=None,
                   unite: str = 'effectifs', diagonale: bool = True):
    """Carte 26 x 26, valeurs dans chaque case, lisible en PNG haute résolution.

    Le fond est une aide visuelle ; les chiffres correspondent aux valeurs réelles.
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    valeurs = matrice.to_numpy(dtype=float).copy()
    if valeurs.shape[0] != valeurs.shape[1]:
        raise ValueError('Une matrice carrée est attendue.')
    if not diagonale:
        np.fill_diagonal(valeurs, np.nan)

    valides = valeurs[np.isfinite(valeurs)]
    if not len(valides):
        raise ValueError('La matrice ne contient aucune valeur exploitable.')

    if unite == 'lift':
        # L'échelle est bornée au 95e percentile pour garder du contraste.
        borne = max(1.25, float(np.percentile(valides, 95)))
        norme = TwoSlopeNorm(vmin=0, vcenter=1, vmax=borne)
        palette = plt.get_cmap('RdBu_r').copy()
        etiquette_couleur = 'Lift (couleurs bornées au 95e percentile)'
    else:
        norme = Normalize(vmin=0, vmax=max(float(valides.max()), 1e-9))
        palette = plt.get_cmap('Blues').copy()
        etiquette_couleur = 'Clients' if unite == 'effectifs' else 'Support (%)'
    palette.set_bad('#e8e8e8')

    n = valeurs.shape[0]
    cote = max(16, 0.62 * n + 2)
    fig, ax = plt.subplots(figsize=(cote + 2.5, cote))
    image = ax.imshow(np.ma.masked_invalid(valeurs), cmap=palette, norm=norme,
                      interpolation='nearest', aspect='equal')
    ax.set_xticks(range(n), labels=matrice.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(n), labels=matrice.index, fontsize=9)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.55)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', length=0, pad=5)
    ax.set_title(titre, fontsize=15, pad=16)
    barre = fig.colorbar(image, ax=ax, shrink=.78, fraction=.035, pad=.025)
    barre.set_label(etiquette_couleur, fontsize=11)

    for i in range(n):
        for j in range(n):
            v = valeurs[i, j]
            if not np.isfinite(v):
                texte = '—' if (i == j and not diagonale) else ''
                contraste = '#666666'
            else:
                if unite == 'effectifs':
                    texte = f'{int(round(v))}'
                elif unite == 'support':
                    texte = f'{v:.1f}'  # pourcentage, mentionné sur la légende
                else:
                    texte = f'{v:.2f}'
                r, g, b, _ = palette(norme(v))
                luminosite = .2126*r + .7152*g + .0722*b
                contraste = '#ffffff' if luminosite < .43 else '#101820'
            ax.text(j, i, texte, ha='center', va='center',
                    fontsize=7.2 if unite == 'effectifs' else 7.5,
                    color=contraste)
    fig.tight_layout()
    if chemin is not None:
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=240, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)


def graphique_matrice_cooccurrences_annotee(effectifs: pd.DataFrame, chemin=None):
    """Co-détention en effectifs, chiffres lisibles dans chaque case."""
    _carte_annotee(effectifs, 'Matrice des cooccurrences — nombre de clients', chemin,
                   unite='effectifs')


def graphique_matrice_supports_annotee(supports: pd.DataFrame, chemin=None):
    """Support joint au format pourcentage (ex. 12,3 = 12,3 %)."""
    _carte_annotee(supports * 100, 'Matrice des supports — en % des clients', chemin,
                   unite='support')


def graphique_matrice_lifts_annotee(lifts: pd.DataFrame, chemin=None):
    """Lifts pour paires distinctes; diagonale non pertinente, donc masquée."""
    _carte_annotee(lifts, 'Matrice des lifts — associations entre produits', chemin,
                   unite='lift', diagonale=False)


def graphique_table_associations(table: pd.DataFrame, chemin=None):
    """Tableau lisible pour rapport, mesures déjà mises en forme."""
    selection = table.copy()
    colonnes = list(selection.columns)
    fig, ax = plt.subplots(figsize=(14, max(3.8, .43 * len(selection) + 1.7)))
    ax.axis('off')
    tableau = ax.table(cellText=selection.astype(str).values,
                       colLabels=colonnes, cellLoc='center', loc='center')
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(9)
    tableau.scale(1.0, 1.55)
    for (r, c), cellule in tableau.get_celld().items():
        cellule.set_edgecolor('#dddddd')
        cellule.set_linewidth(.5)
        if r == 0:
            cellule.set_facecolor('#e9eef5')
            cellule.set_text_props(weight='bold')
        elif r % 2 == 0:
            cellule.set_facecolor('#f7f8fa')
    ax.set_title('Associations à examiner — sélection exploratoire', fontsize=13, pad=15)
    fig.tight_layout()
    if chemin is not None:
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin, dpi=240, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)
