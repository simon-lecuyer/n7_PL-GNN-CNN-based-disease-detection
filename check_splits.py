import json

for model_type in ['cnn', 'gnn']:
    print(f'\n{"="*50}')
    print(f'{model_type.upper()} - Vérification des simulations')
    print('='*50)
    
    base = 'data/processed/processed_20260305_121946/datasets'
    
    # Charger les splits
    with open(f'{base}/{model_type}/train.json') as f:
        train_data = json.load(f)
    with open(f'{base}/{model_type}/val.json') as f:
        val_data = json.load(f)
    with open(f'{base}/{model_type}/test.json') as f:
        test_data = json.load(f)
    
    # Accéder à la liste 'samples' dans chaque fichier
    train = train_data['samples']
    val = val_data['samples']
    test = test_data['samples']
    
    # Extraire les simulation_id uniques
    train_sims = set(item['sim_id'] for item in train)
    val_sims = set(item['sim_id'] for item in val)
    test_sims = set(item['sim_id'] for item in test)
    
    print(f'\nTrain: {len(train_sims)} simulations uniques')
    print(f'  IDs: {sorted(train_sims)}')
    print(f'\nVal: {len(val_sims)} simulations uniques')
    print(f'  IDs: {sorted(val_sims)}')
    print(f'\nTest: {len(test_sims)} simulations uniques')
    print(f'  IDs: {sorted(test_sims)}')
    
    # Vérifier les intersections
    train_val = train_sims & val_sims
    train_test = train_sims & test_sims
    val_test = val_sims & test_sims
    
    print(f'\n{"="*50}')
    print('VÉRIFICATION DES CHEVAUCHEMENTS')
    print('='*50)
    print(f'Train ∩ Val: {train_val if train_val else "✓ Aucun"}')
    print(f'Train ∩ Test: {train_test if train_test else "✓ Aucun"}')
    print(f'Val ∩ Test: {val_test if val_test else "✓ Aucun"}')
    
    # Vérification complète
    all_sims = train_sims | val_sims | test_sims
    print(f'\nTotal simulations dans tous les splits: {len(all_sims)}')
    print(f'Total échantillons: train={len(train)}, val={len(val)}, test={len(test)}')
    
    if not (train_val or train_test or val_test):
        print(f'\n PAS DE CHEVAUCHEMENT - Les splits sont propres pour {model_type.upper()}!')
    else:
        print(f'\n ATTENTION: Chevauchement détecté pour {model_type.upper()}!')
