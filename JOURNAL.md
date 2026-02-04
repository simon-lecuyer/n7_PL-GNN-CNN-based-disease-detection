# 📝 Journal de Développement - N7 Projet Long

> **Projet :** GNN-CNN Based Disease Detection in Agriculture  
> **Équipe :** Myriam ROBBANA, Mihai COSTIN, Simon LECUYER, Assala ASSELALOU, Yassin MOUKAN  
> **Période :** Février 2026 - Mars 2026

---

## 📋 Instructions d'utilisation

Chaque participant ajoute une entrée pour documenter son travail. Format requis :

```markdown
### [Date] - [Heure] - [Prénom NOM]
**Branche :** `nom-de-la-branche`  
**Tâche :** Description courte de la tâche  
**Modifications :**
- Point 1
- Point 2

**Résultats/Observations :** Ce qui a été appris ou observé  
**Problèmes rencontrés :** (si applicable)  
**Prochaines étapes :** Ce qui reste à faire
```

---

## 📅 Entrées du Journal

### [03/02/2026] - [15:30] - Simon LECUYER
**Branche :** `main`  
**Tâche :** Initialisation de la structure du projet

**Modifications :**
- Création de l'architecture de dossiers (data/, gnn/, cnn/, results/, configs/, utils/, scripts/)
- Rédaction du README.md avec documentation complète
- Configuration du .gitignore pour PyTorch
- Création du requirements.txt avec toutes les dépendances

**Résultats/Observations :**
- Structure modulaire et propre prête pour le développement

**Problèmes rencontrés :** Aucun

**Prochaines étapes :**
- Génération des premières données avec WaterberryFarms
- Implémentation des scripts de base dans scripts/

---

### [04/02/2026] - [16:00] - Simon LECUYER
**Branche :** `main`  
**Tâche :** Intégration de WaterberryFarms et création du script de génération de données

**Modifications :**
- **Création de `scripts/generate_data.py`** : Script complet de génération de données
  - Import automatique de WaterberryFarms depuis le dossier parent
  - Support des modèles épidémique (SIR) et dissipation
  - Génération de données au format images (CNN) et graphes (GNN)
  - Tous les paramètres configurables via ligne de commande
  - Métadonnées JSON pour chaque génération et simulation
  - Organisation automatique : dossier daté avec sous-dossiers par simulation
  
- **Mise à jour du README.md** :
  - Section "Prérequis WaterberryFarms" expliquant la structure de dossiers requise
  - Documentation complète des paramètres de `generate_data.py`
  - Tableau récapitulatif des paramètres du modèle épidémique
  - Exemples d'utilisation pour différents cas (test, entraînement, formats spécifiques)
  - Structure de sortie détaillée des données générées

**Résultats/Observations :**
- ✅ WaterberryFarms utilisé comme bibliothèque (pas de sous-module Git)
- ✅ Détection automatique du chemin vers WaterberryFarms
- ✅ Génération double format par défaut : images PNG pour CNN + graphes NumPy pour GNN
- ✅ Modèle SIR (épidémique) par défaut avec option de basculer vers dissipation
- ✅ Organisation propre : `data/simulations/generation_TIMESTAMP/sim_XXXX/{images,graphs}/`
- ✅ Métadonnées complètes pour traçabilité (paramètres, seed, timestamp)

**Architecture de données générées :**
```
data/simulations/generation_20260204_160000/
├── generation_metadata.json          # Paramètres globaux
├── sim_0000/
│   ├── metadata.json                 # Paramètres spécifiques
│   ├── images/                       # Pour CNN
│   │   └── t_XXXX.png
│   └── graphs/                       # Pour GNN
│       └── t_XXXX.npy (nodes, edges, features)
```

**Problèmes rencontrés :** 
- Aucun - import relatif fonctionne correctement avec la structure actuelle

**Prochaines étapes :**
- Tester la génération avec différents paramètres
- Créer `utils/data_loader.py` pour charger les données générées
- Implémenter `utils/graph_builder.py` pour construire les graphes PyTorch Geometric
- Commencer l'architecture CNN baseline

**Fichiers créés :**
1. `scripts/generate_data.py` - Script principal de génération
2. `configs/data_generation_example.txt` - Configuration exemple
3. `data/simulations/README.md` - Documentation des données

**Résumé des fonctionnalités :**
- ✅ Import automatique de WaterberryFarms
- ✅ Génération double format (images + graphes)
- ✅ Métadonnées JSON complètes
- ✅ Organisation propre des données générées
- ✅ Configuration via ligne de commande ou fichier texte

---

<!-- Ajoutez vos entrées ci-dessous en respectant le format -->

### [JJ/MM/AAAA] - [HH:MM] - Prénom NOM
**Branche :** `votre-branche`  
**Tâche :** Description

**Modifications :**
- ...

**Résultats/Observations :** ...

**Problèmes rencontrés :** ...

**Prochaines étapes :** ...

---

## 🎯 Objectifs par Semaine

### ✅ Semaine 1 : Data Preparation (03-09/02/2026)
- [ ] **Team 1 (CNN)** : Générer datasets d'images
- [ ] **Team 2 (GNN)** : Extraire structure de graphe
- [ ] **Commun** : Documentation des spécifications

### ⏳ Semaine 2 : Baseline Models (10-16/02/2026)
- [ ] **Team 1** : Architecture CNN baseline
- [ ] **Team 2** : Structure graphe GNN + baseline

### ⏳ Semaine 3 : Optimization & CGP (17-23/02/2026)
- [ ] **Team 1** : Optimisation CNN + inférence temps réel
- [ ] **Team 2** : Amélioration GNN + intégration CGP

### ⏳ Semaine 4 : Evaluation (24/02-02/03/2026)
- [ ] Tests comparatifs CNN vs GNN
- [ ] Benchmarks sur différents scenarios

### ⏳ Semaine 5 : Finalization (03-09/03/2026)
- [ ] Finalisation des modèles
- [ ] Rédaction rapport final
- [ ] Préparation présentation

---

## 💡 Bonnes Pratiques

1. **Branches :** Créer une branche par fonctionnalité (`feature/nom-feature`)
2. **Commits :** Messages clairs et descriptifs
3. **Pull Requests :** Code review avant merge dans `main`
4. **Tests :** Tester avant de commit
5. **Documentation :** Commenter le code complexe

---

## 🐛 Problèmes Récurrents

*(À remplir au fur et à mesure)*

---

**Dernière mise à jour :** 03/02/2026 - Simon LECUYER
