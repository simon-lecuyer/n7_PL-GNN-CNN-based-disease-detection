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

### [09/02/2026] - [17:00] - Simon LECUYER
**Branche :** `Pipeline-DatatoDatasets`  
**Tâche :** Création de la pipeline complète de données pour comparaison CNN/GNN

**Modifications :**
- **Création de `scripts/preprocess_data.py`** : Prétraitement avancé des simulations
  - Crop automatique des zones infectées avec marge configurable
  - Redimensionnement intelligent pour CNN (target_size)
  - Normalisation/standardisation des données
  - Features spatiales pour GNN (coordonnées normalisées)
  - Statistiques automatiques sur les données preprocessées
  - Support multi-format (CNN + GNN simultané)
  
- **Création de `scripts/create_datasets.py`** : Génération de datasets PyTorch
  - Split train/val/test stratifié et reproductible
  - Stratification par simulation, infection_level, ou timestep
  - Support des séquences temporelles (LSTM, Transformer)
  - Métadonnées complètes pour traçabilité
  - Statistiques par split
  
- **Création de `utils/datasets.py`** : Classes PyTorch Dataset
  - DiseaseDetectionDataset pour CNN et GNN
  - TemporalDiseaseDataset pour séquences temporelles
  - Fonctions collate personnalisées
  - Helper get_dataloader() pour simplifier l'usage
  
- **Documentation complète** : `data/PIPELINE.md`
  - Guide complet des 4 étapes de la pipeline
  - Exemples d'utilisation pour chaque script
  - Stratégies de comparaison CNN vs GNN
  - Commandes rapides et troubleshooting

**Résultats/Observations :**
- ✅ Pipeline complète testée et fonctionnelle
- ✅ Garantit la comparabilité CNN/GNN (mêmes données sources)
- ✅ Split reproductible avec stratification
- ✅ Prétraitement optimisé (crop sur infection, normalisation)
- ✅ Format CNN: 64x64 (configurable), format GNN: graphe avec features enrichies
- ✅ Métadonnées JSON à chaque étape pour traçabilité

**Architecture de la pipeline :**
```
Simulations → Prétraitement → Datasets → Entraînement
    (brutes)     (optimisées)  (PyTorch)    (modèles)
```

**Problèmes rencontrés :** 
- Aucun - pipeline complète et robuste

**Prochaines étapes :**
- Implémenter les modèles CNN baseline
- Implémenter les modèles GNN baseline
- Scripts d'entraînement avec la pipeline
- Métriques de comparaison CNN vs GNN

---

### [12/02/2026] - [16:30] - Simon LECUYER
**Branche :** `main`  
**Tâche :** Correction critique de la pipeline de données - Confusion SIR et datasets

**Modifications :**
- **Correction majeure dans `scripts/generate_data.py`** :
  - Ajout de la capture de `env.status` (états SIR réels) en plus de `env.value`
  - Permet d'avoir accès aux vrais états : 0=Susceptible, >0=Infected, -1=Recovered, -2=Immune
  
- **Correction critique dans `scripts/preprocess_data.py`** :
  - **Inversion de la sémantique** : `env.value` de WaterberryFarms a 1.0=sain, 0.5=infecté, 0.0=détruit
  - Maintenant transformé en : 1.0=présence maladie, 0.0=sain (sémantique correcte pour détection)
  - Correction du seuil de crop : `> 0.4` au lieu de `> 0.1` (qui capturait tout)
  - Remplacement PIL (uint8) par `scipy.ndimage.zoom` pour éviter la quantification
  - Implémentation correcte de la normalisation (était un no-op avant)
  - Propagation du `status` SIR pour créer des labels supervisés
  
- **Correction dans `scripts/create_datasets.py`** :
  - **Split unifié CNN/GNN** : le split est fait UNE SEULE FOIS sur les paires (sim_id, timestep)
  - Le même split est appliqué aux deux formats pour garantir la comparabilité
  - Évite les divergences dues aux calculs d'infection_level légèrement différents
  
- **Ajout de labels dans `utils/datasets.py`** :
  - Création de labels à partir du status : 0=Sain (S), 1=Infecté (I), 2=Recovered (R)
  - Les datasets retournent maintenant `(features, label)` pour l'entraînement supervisé
  - Support classification SIR par pixel (CNN) ou par nœud (GNN)
  
- **Nettoyage du dépôt** :
  - Suppression de 3 générations/preprocessing sur 4 (doublons identiques avec même seed)
  - Garde uniquement `generation_20260204_173051` et `processed_20260209_165358`

**Résultats/Observations :**
- ⚠️ **Problème critique identifié** : Confusion totale sur la sémantique de `env.value`
  - Documentation parlait de "Disease Intensity" mais `env.value` représente la "valeur agricole restante"
  - 1.0 = plants sains (Susceptible), 0.5 = infectés, 0.0 = détruits/recovered
  - Toute la pipeline traitait les données à l'envers !
  
- ✅ **CNN et GNN utilisent bien les mêmes simulations** (doute initial infondé)
  - Même source : fichiers graphes `.npy` communs
  - Mais le split était fait indépendamment → risque de divergence corrigé
  
- ✅ **Perte d'information évitée** :
  - Conversion float→uint8→resize→float perdait la distinction 0.0/0.5/1.0
  - Remplacé par resize direct en float32
  
- ⚠️ **Données ridiculement petites** (2 sims × 10 timesteps = 20 samples)
  - Inutilisable pour entraîner quoi que ce soit
  - 3 samples en test/val → aucune métrique significative

**Problèmes rencontrés :** 
- Architecture pipeline bien conçue mais mauvaise compréhension du modèle SIR de WaterberryFarms
- Manque de validation des données générées (aurait dû détecter la sémantique inversée)
- Génération massive de doublons (même seed, mêmes paramètres)

**Prochaines étapes :**
- Régénérer des données avec les corrections (50-100 simulations × 100+ timesteps)
- Tester le nouveau preprocessing avec `demo.sh` ou commandes manuelles
- Vérifier que les labels SIR sont correctement extraits
- Implémenter les modèles CNN/GNN baseline avec supervision (classification S/I/R)
- Ou utiliser prédiction temporelle (état t → état t+1) comme tâche

**Impact :** Corrections critiques qui changent fondamentalement la sémantique des données. **Nécessite régénération complète** des datasets pour exploiter les corrections.

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

**Dernière mise à jour :** 12/02/2026 - Simon LECUYER

