# Projet Data Mining – Master SIAD

Étude des associations entre produits bancaires pour proposer des pistes de *cross-selling* sur l'assurance et l'épargne.

## Organisation

```text
.
├── data/                         # données.csv
├── notebooks/
│   ├── 01_audit_qualite_donnees.ipynb
│   └── 02_preparation_donnees.ipynb
├── src/
│   ├── __init__.py
│   ├── chargement.py             # lecture du CSV et colonnes produits
│   └── preparation.py            # scénarios de préparation
├── tests/
│   └── test_preparation.py
├── requirements.txt
└── .gitignore
```

Les notebooks présentent les questions, résultats et interprétations. Les fonctions réutilisables sont dans `src/`.

## Environnement de développement

- Python 3.11
- Environnement virtuel : `.venv`
- Dépendances : `requirements.txt`

## Configuration de l'environnement (chaque membre de l'équipe)

Prérequis : **Python 3.11** (version utilisée pour la vérification de ce kit), Git.

Depuis la racine du dépôt :

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dm-siad --display-name "Python (Data Mining SIAD)"
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name dm-siad --display-name "Python (Data Mining SIAD)"
```

Dans VS Code, choisir l'interpréteur `.venv` et le noyau **Python (Data Mining SIAD)**. Pour ouvrir Jupyter : `python -m jupyter lab`.

**Attention :** on ne partage pas `.venv` dans Git (dépendances et chemins propres à chaque ordinateur). C'est `requirements.txt` qui décrit les versions communes ; Python 3.13 doit aussi être utilisé par tous.

### Données

Copier le CSV original dans `data/données.csv` (séparateur `;`). Ne pas le modifier et ne pas le committer. Les notebooks fonctionnent depuis `notebooks/` ou depuis la racine.

### Vérification rapide

```bash
python -m unittest discover -s tests -v
```

## Hypothèses provisoires

- **Scénario principal :** segment précédant la première rupture décroissante d'identifiants, une ligne par ID.
- **Scénario complémentaire :** données complètes regroupées par ID ; `max` sur les produits binaires représente la présence observée au moins une fois.
- `SEXE` est conservé dans le fichier d'origine mais ne participe pas à la matrice d'association.

Ces choix ne prouvent pas que le segment final est erroné ni que les répétitions correspondent à une même personne. Leur effet est comparé de manière descriptive dans le notebook 02.

## Périmètre du cours

Les prochaines analyses reprendront la démarche du document *Cours associations* : construction de la matrice d'association et calcul du support, de la confiance, de la confiance attendue et du lift. La préparation ne modifie jamais le CSV original.
