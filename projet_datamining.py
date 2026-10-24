"""
Projet data mining : nettoyage, exploration, règles d'association, clustering.

Fonctionne de deux façons, sans rien modifier :

1) En terminal (depuis la racine du projet, .venv activé) :
       pip install pandas numpy scipy scikit-learn matplotlib seaborn mlxtend
       python projet_datamining.py
   Options : --data chemin.csv --table chemin_table.csv --out dossier

2) Dans un notebook Jupyter : coller/importer ce fichier dans une cellule, puis
   exécuter dans la cellule suivante :
       main()

Les chemins sont détectés automatiquement (data/ ou ../data/).
Les résultats (CSV + graphiques) sont écrits dans outputs/ à côté de data/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

EN_NOTEBOOK = "ipykernel" in sys.modules
if not EN_NOTEBOOK:
    matplotlib.use("Agg")  # en terminal : on enregistre les figures sans les afficher

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

COL_ID = "id"
COL_SEXE = "SEXE"
SEED = 42

# Produits exclus des règles et du clustering :
#  - DAV : détenu par ~99 % des clients, n'apporte aucune information
#  - ASSPREVDECDEP : agrégat exact de ASSPREVDEC OU ASSDEP (redondant)
EXCLURE = ["DAV", "ASSPREVDECDEP"]

# Règles d'association
MIN_SUPPORT = 0.05
MIN_CONFIANCE = 0.60
MIN_LIFT = 1.30
TAILLE_MAX_ITEMSET = 4        # antécédent + conséquent <= 4 produits (règles lisibles)

# Clustering
K_MIN, K_MAX = 3, 8
K_FINAL = None                # None = k choisi automatiquement (meilleure silhouette)


def titre(txt):
    print(f"\n{'=' * 70}\n{txt}\n{'=' * 70}")


def finir_figure(fig, chemin):
    """Enregistre la figure ; l'affiche dans un notebook, la ferme en terminal."""
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    if EN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# 0. Libellés
# --------------------------------------------------------------------------
def charger_libelles(chemin, colonnes_df):
    """Dictionnaire {colonne du dataset: intitulé} depuis la table des codes produits."""
    t = pd.read_csv(chemin, sep=";", encoding="utf-8-sig")
    t.columns = ["code", "libelle"]
    t["code"] = t["code"].astype(str).str.strip()
    t["libelle"] = t["libelle"].astype(str).str.strip()
    lib = {c.upper(): l for c, l in zip(t["code"], t["libelle"])}

    cols = list(colonnes_df)
    manquants = [c for c in cols if c.upper() not in lib]
    inconnus = [c for c in t["code"] if c.upper() not in {x.upper() for x in cols}]
    print(f"Table des codes : {len(t)} lignes (inclut SEXE et id)")
    print(f"  Colonnes du dataset sans libellé : {manquants or 'aucune'}")
    print(f"  Codes de la table absents du dataset : {inconnus or 'aucun'}")
    return {c: lib[c.upper()] for c in cols if c.upper() in lib}


# --------------------------------------------------------------------------
# 1. Nettoyage
# --------------------------------------------------------------------------
def nettoyer(df, out):
    titre("1. NETTOYAGE")
    print(f"Dimensions brutes : {df.shape}")

    n_dup = int(df.duplicated().sum())
    print(f"Lignes entièrement dupliquées (supprimées) : {n_dup}")
    df = df.drop_duplicates().copy()

    # Même id mais contenu différent : ce sont des clients distincts -> on les garde, on les signale
    mask = df.duplicated(COL_ID, keep=False)
    print(f"Lignes dont l'id est partagé avec un autre client (conservées) : {int(mask.sum())}")
    if mask.any():
        df[mask].sort_values(COL_ID).to_csv(out / "anomalies_id_dupliques.csv", index=False)
        print("  -> détail dans anomalies_id_dupliques.csv ; l'id n'est donc pas une clé fiable")
    df = df.reset_index(drop=True)
    df.insert(0, "ligne", df.index)  # clé technique unique

    n_na = int(df[COL_SEXE].isna().sum())
    df[COL_SEXE] = df[COL_SEXE].fillna("Inconnu")
    print(f"SEXE manquant remplacé par 'Inconnu' : {n_na} clients")

    produits = [c for c in df.columns if c not in ("ligne", COL_ID, COL_SEXE)]
    df[produits] = df[produits].astype(int)
    assert set(np.unique(df[produits].values)) <= {0, 1}, "Produits non binaires !"

    taux = df[produits].mean()
    print("Produits détenus à plus de 95 % :", list(taux[taux > 0.95].index))
    print(f"Dimensions après nettoyage : {df.shape}")

    df.to_csv(out / "donnees_nettoyees.csv", index=False, sep=";")
    return df, produits


# --------------------------------------------------------------------------
# 2. Exploration
# --------------------------------------------------------------------------
def jaccard_matrix(X):
    inter = X.T @ X
    s = np.diag(inter)
    union = s[:, None] + s[None, :] - inter
    return inter / np.where(union == 0, 1, union)


def explorer(df, produits, libelles, out):
    titre("2. EXPLORATION : DETENTION, CO-DETENTION, SEXE")

    # Taux de détention avec libellés
    taux = df[produits].mean().mul(100).sort_values(ascending=False)
    tab = pd.DataFrame({
        "libelle": [libelles.get(p, "") for p in taux.index],
        "effectif": df[produits].sum().reindex(taux.index),
        "taux_detention_%": taux.round(2),
    })
    tab.to_csv(out / "taux_detention.csv")

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(taux.index[::-1], taux.values[::-1], color="steelblue")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=2)
    ax.set(xlabel="% de clients", title="Taux de détention par produit")
    plt.tight_layout()
    finir_figure(fig, out / "taux_detention.png")

    nb = df[produits].sum(axis=1)
    print(f"Produits par client : moyenne {nb.mean():.2f}, médiane {nb.median():.0f}")

    # Corrélation / Jaccard
    X = df[produits].values.astype(float)
    corr = pd.DataFrame(np.corrcoef(X.T), index=produits, columns=produits)
    jac = pd.DataFrame(jaccard_matrix(X), index=produits, columns=produits)
    corr.round(3).to_csv(out / "correlation_produits.csv")
    jac.round(3).to_csv(out / "jaccard_produits.csv")

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[0],
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": .7})
    axes[0].set_title("Corrélation entre produits")
    sns.heatmap(jac, cmap="viridis", vmin=0, vmax=1, ax=axes[1],
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": .7})
    axes[1].set_title("Similarité de Jaccard entre produits")
    plt.tight_layout()
    finir_figure(fig, out / "heatmaps_produits.png")

    paires = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    top = paires.reindex(paires.abs().sort_values(ascending=False).index).head(15)
    top.rename("correlation").to_csv(out / "top_paires_correlees.csv")
    print("\nTop 10 des paires les plus corrélées :")
    print(top.head(10).round(3).to_string())

    # Redondances logiques exactes
    print("\nRedondances logiques :")
    if {"ASSPREVDEC", "ASSDEP", "ASSPREVDECDEP"} <= set(produits):
        ok = (((df.ASSPREVDEC == 1) | (df.ASSDEP == 1)).astype(int) == df.ASSPREVDECDEP).mean()
        print(f"  ASSPREVDECDEP = ASSPREVDEC OU ASSDEP : vrai pour {ok:.1%} des clients")
    if {"CART", "CPAI"} <= set(produits):
        n = int(((df.CPAI == 1) & (df.CART == 0)).sum())
        print(f"  Clients avec CPAI sans CART : {n}  (CPAI est quasi inclus dans CART)")

    # Sexe : taux + khi-deux
    sous = df[df[COL_SEXE].isin(["F", "M"])]
    lignes = []
    for p in produits:
        t = pd.crosstab(sous[COL_SEXE], sous[p])
        pval = chi2_contingency(t)[1] if t.shape == (2, 2) else np.nan
        lignes.append({
            "produit": p,
            "libelle": libelles.get(p, ""),
            "taux_F_%": round(sous.loc[sous[COL_SEXE] == "F", p].mean() * 100, 1),
            "taux_M_%": round(sous.loc[sous[COL_SEXE] == "M", p].mean() * 100, 1),
            "p_value": pval,
        })
    sexe = pd.DataFrame(lignes)
    sexe["ecart_M_moins_F_pts"] = (sexe["taux_M_%"] - sexe["taux_F_%"]).round(1)
    sexe = sexe.reindex(sexe["ecart_M_moins_F_pts"].abs().sort_values(ascending=False).index)
    sexe.to_csv(out / "detention_par_sexe_tests.csv", index=False)
    print("\nÉcarts F/M les plus marqués (hors 'Inconnu') :")
    print(sexe.head(8)[["produit", "taux_F_%", "taux_M_%", "ecart_M_moins_F_pts"]].to_string(index=False))


# --------------------------------------------------------------------------
# 3. Règles d'association
# --------------------------------------------------------------------------
def regles_association(df, produits, libelles, out):
    titre("3. REGLES D'ASSOCIATION (FP-Growth)")
    try:
        from mlxtend.frequent_patterns import association_rules, fpgrowth
    except ImportError:
        print("mlxtend non installé (pip install mlxtend) : étape ignorée.")
        return

    prods = [p for p in produits if p not in EXCLURE]
    print(f"Produits exclus : {EXCLURE} | produits analysés : {len(prods)}")
    basket = df[prods].astype(bool)

    freq = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True, max_len=TAILLE_MAX_ITEMSET)
    print(f"Itemsets fréquents (support >= {MIN_SUPPORT}, taille <= {TAILLE_MAX_ITEMSET}) : {len(freq)}")

    try:
        regles = association_rules(freq, num_itemsets=len(basket),
                                   metric="confidence", min_threshold=MIN_CONFIANCE)
    except TypeError:  # anciennes versions de mlxtend
        regles = association_rules(freq, metric="confidence", min_threshold=MIN_CONFIANCE)

    regles = regles[regles["lift"] >= MIN_LIFT].copy()
    regles = regles.sort_values(["lift", "confidence"], ascending=False)

    fmt = lambda s: ", ".join(sorted(s))
    regles["antecedent"] = regles["antecedents"].apply(fmt)
    regles["consequent"] = regles["consequents"].apply(fmt)
    regles["antecedent_libelle"] = regles["antecedents"].apply(
        lambda s: " + ".join(libelles.get(p, p) for p in sorted(s)))
    regles["consequent_libelle"] = regles["consequents"].apply(
        lambda s: " + ".join(libelles.get(p, p) for p in sorted(s)))

    cols = ["antecedent", "consequent", "support", "confidence", "lift", "leverage",
            "conviction", "antecedent_libelle", "consequent_libelle"]
    res = regles[cols].round(4)
    res.to_csv(out / "regles_association.csv", index=False)
    print(f"Règles retenues (confiance >= {MIN_CONFIANCE}, lift >= {MIN_LIFT}) : {len(res)}")
    if res.empty:
        print("Aucune règle : baisser MIN_SUPPORT / MIN_LIFT / MIN_CONFIANCE.")
        return
    print("\nTop 15 par lift :")
    print(res.head(15)[["antecedent", "consequent", "support", "confidence", "lift"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(res["support"], res["confidence"], c=res["lift"], cmap="plasma", s=30, alpha=.8)
    plt.colorbar(sc, label="lift")
    ax.set(xlabel="Support", ylabel="Confiance", title="Règles d'association : support / confiance / lift")
    plt.tight_layout()
    finir_figure(fig, out / "regles_scatter.png")


# --------------------------------------------------------------------------
# 4. Segmentation
# --------------------------------------------------------------------------
def clustering(df, produits, libelles, out):
    titre("4. SEGMENTATION (SVD + K-Means)")
    prods = [p for p in produits if p not in EXCLURE]
    X = df[prods].values.astype(float)

    Z = TruncatedSVD(n_components=min(10, len(prods) - 1), random_state=SEED).fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), size=min(10000, len(Z)), replace=False)

    scores, inerties = {}, {}
    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        scores[k] = silhouette_score(Z[idx], km.labels_[idx])
        inerties[k] = km.inertia_
        print(f"  k={k}  silhouette={scores[k]:.3f}  inertie={inerties[k]:,.0f}")

    k_best = K_FINAL or max(scores, key=scores.get)
    print(f"-> k retenu : {k_best}  (silhouette faible = clusters peu séparés, à nuancer dans le rapport)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(inerties), list(inerties.values()), "o-")
    axes[0].set(title="Méthode du coude", xlabel="k", ylabel="Inertie")
    axes[1].plot(list(scores), list(scores.values()), "o-", color="darkorange")
    axes[1].set(title="Silhouette", xlabel="k", ylabel="Score")
    plt.tight_layout()
    finir_figure(fig, out / "clustering_choix_k.png")

    labels = KMeans(n_clusters=k_best, n_init=10, random_state=SEED).fit_predict(Z)
    d = df.copy()
    d["cluster"] = labels
    taille = d["cluster"].value_counts().sort_index()

    profil = d.groupby("cluster")[prods].mean().mul(100).round(1)
    ecart = profil.sub(d[prods].mean().mul(100), axis=1).round(1)
    profil["nb_produits_moy"] = (d.groupby("cluster")[prods].sum().sum(axis=1) / taille).round(2)
    profil["%_hommes"] = (d.groupby("cluster")[COL_SEXE].apply(lambda s: (s == "M").mean()) * 100).round(1)
    profil["effectif"] = taille
    profil["%_clients"] = (taille / len(d) * 100).round(1)
    profil.to_csv(out / "profil_clusters.csv")

    print("\nProfil des clusters (écart à la moyenne, en points de %) :")
    for c in ecart.index:
        s = ecart.loc[c].sort_values(ascending=False)
        plus = ", ".join(f"{p} ({v:+.1f})" for p, v in s.head(4).items())
        moins = ", ".join(f"{p} ({v:+.1f})" for p, v in s.tail(2).items())
        print(f"  Cluster {c} - {profil.loc[c, '%_clients']}% des clients, "
              f"{profil.loc[c, 'nb_produits_moy']} produits en moyenne")
        print(f"      surreprésenté : {plus}")
        print(f"      sous-représenté : {moins}")

    fig, ax = plt.subplots(figsize=(14, 5 + k_best * .4))
    sns.heatmap(ecart, cmap="RdBu_r", center=0, annot=True, fmt=".0f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Écart de détention par cluster vs moyenne (points de %)")
    plt.tight_layout()
    finir_figure(fig, out / "clusters_profil_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z[idx, 0], Z[idx, 1], c=labels[idx], cmap="tab10", s=6, alpha=.5)
    ax.set(title="Clusters (projection SVD, échantillon)", xlabel="Axe 1", ylabel="Axe 2")
    plt.tight_layout()
    finir_figure(fig, out / "clusters_projection.png")

    d[["ligne", COL_ID, "cluster"]].to_csv(out / "clients_clusters.csv", index=False)


# --------------------------------------------------------------------------
def trouver(candidats):
    for c in candidats:
        if Path(c).exists():
            return Path(c)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default=None)
    # parse_known_args : ignore les arguments que Jupyter ajoute (--f=...json)
    args, _ = ap.parse_known_args(argv)

    data = Path(args.data) if args.data else trouver(
        ["data/données.csv", "../data/données.csv", "données.csv"])
    if data is None or not data.exists():
        raise FileNotFoundError("données.csv introuvable : utiliser --data chemin/vers/données.csv")
    table = Path(args.table) if args.table else trouver(
        [data.parent / "données_table.csv", "data/données_table.csv", "../data/données_table.csv"])
    out = Path(args.out) if args.out else data.parent.parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(data, sep=";")
    if COL_ID not in df.columns:
        raise ValueError(
            f"Colonne '{COL_ID}' absente (colonnes lues : {list(df.columns)[:6]}...). "
            "Le fichier n'est probablement pas le bon : 'données.csv' doit contenir les clients "
            "(73 930 lignes), pas la table des codes produits. Restaurer avec : git restore data/données.csv"
        )
    print(f"Données : {data} {df.shape}")

    libelles = {}
    if table is not None and Path(table).exists():
        titre("0. LIBELLES PRODUITS")
        libelles = charger_libelles(table, [c for c in df.columns if c not in (COL_ID, COL_SEXE)])
    else:
        print("Table des codes introuvable : libellés ignorés.")

    df, produits = nettoyer(df, out)
    explorer(df, produits, libelles, out)
    regles_association(df, produits, libelles, out)
    clustering(df, produits, libelles, out)

    titre("TERMINE")
    print(f"Résultats dans : {out.resolve()}")
    return df


if __name__ == "__main__" and not EN_NOTEBOOK:
    main()
