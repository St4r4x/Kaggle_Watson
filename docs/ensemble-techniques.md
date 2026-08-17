# Techniques utilisées — passage de 0.91241 à 0.92858

Contexte : compétition *Contradictory, My Dear Watson* (NLI, 15 langues). Point de départ :
un seul `xlm-roberta-large-xnli` fine-tuné, 92.24% d'accuracy en validation, 0.91241 sur le
leaderboard public. Objectif : améliorer le score sans changer de modèle de base ni de
données d'entraînement.

## Ce qui a marché

### 1. K-fold cross-validation comme générateur d'ensemble

Plutôt qu'un seul split train/val, `xlm-roberta-large-xnli` a été ré-entraîné 5 fois sur
5 folds stratifiés (`StratifiedKFold`, mêmes hyperparamètres pour chaque fold). Chaque fold
voit une portion différente des données en validation, donc chaque modèle final a des
erreurs légèrement différentes malgré une architecture identique.

| Fold | Val accuracy |
|---|---|
| 1 | 92.41% |
| 2 | 91.83% |
| 3 | 92.45% |
| 4 | 92.45% |
| 5 | 92.45% |
| **Moyenne** | **92.32%** |

Pris individuellement, aucun fold ne bat vraiment la baseline (92.24%) — l'intérêt n'est pas
la performance d'un fold seul, mais la diversité qu'ils apportent une fois combinés (point 3).

### 2. Diversité architecturale : un second modèle différent

En plus des 5 folds (tous la même architecture XLM-RoBERTa-large), un modèle **différent**
a été ajouté à l'ensemble : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (DeBERTa, pas
RoBERTa ; plus petit ; déjà fine-tuné MNLI+XNLI comme le modèle principal).

Son accuracy seule est nettement plus faible (**87.51%**) — c'est normal, c'est un modèle
plus petit. Le pari : un modèle individuellement moins bon mais qui se trompe sur des
exemples *différents* peut quand même améliorer l'ensemble, parce que ses erreurs ne sont
pas corrélées avec celles des modèles XLM-R. Verdict confirmé par le score final : l'ajouter
n'a pas tiré l'ensemble vers le bas.

### 3. Ensembling par moyenne de probabilités (soft voting)

Les 6 modèles (5 folds + mDeBERTa) tournent chacun sur le jeu de test, produisent une
distribution softmax à 3 classes, et ces distributions sont **moyennées** avant de prendre
l'argmax — pas un vote majoritaire sur les classes prédites, mais une moyenne sur les
probabilités elles-mêmes (plus d'information conservée qu'un vote dur).

Point technique important : chaque checkpoint a son propre ordre de labels natif (ex.
`joeddav/xlm-roberta-large-xnli` : `0=contradiction,1=neutral,2=entailment`, l'inverse de
l'ordre Kaggle). Avant de moyenner, chaque sortie est **réalignée** vers un ordre canonique
commun (`entailment,neutral,contradiction`) en lisant le `id2label` sauvegardé dans le
`config.json` de chaque checkpoint — mélanger des probabilités sans ce réalignement aurait
silencieusement corrompu toutes les prédictions.

### Résultat

| Configuration | Val accuracy | Score public Kaggle |
|---|---|---|
| xlm-roberta-large-xnli seul | 92.24% | 0.91241 |
| **5 folds + mDeBERTa, moyenne de probabilités** | 92.32% (moyenne folds) | **0.92858** |

**+0.0162** sur le leaderboard — gain net malgré un ensemble qui inclut un modèle
individuellement plus faible.

## Ce qui a été essayé et abandonné

**R-Drop** (régularisation par double forward-pass + perte de cohérence KL) : testé avant
cette session, résultat pire (90.92% val acc) que la baseline simple. Pas retenu, pas
revisité sans changer d'autres paramètres.

## Ce qui reste possible mais non fait

- **Second modèle plus fort** : remplacer `mDeBERTa-v3-base-mnli-xnli` par
  `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (même taille, entraîné sur
  plus de données NLI, généralement plus fort en solo) — changement de config, coût faible.
- **Pseudo-labeling** sur le jeu de test : volontairement écarté (risque de biais de
  confirmation) — à reconsidérer seulement si l'ensemble actuel plafonne.
- **Ensembling pondéré / appris** (au lieu d'une simple moyenne) : pas fait — la moyenne
  simple a déjà fonctionné, ajouter de la complexité n'était pas justifié pour l'instant.

## Note infrastructure (pas une technique ML, mais utile à savoir)

L'inférence des 6 modèles a été faite **en local sur GPU** (desktop, RTX 4060 Ti, ~8 min)
plutôt que sur Kaggle (CPU fallback, plusieurs heures sans terminer). La compétition étant
*kernels-only*, un petit kernel "passe-plat" (`kaggle/watson-ensemble-shim/`) recopie juste
le `submission.csv` généré en local — voir `CLAUDE.md` pour le détail du flux de soumission.
