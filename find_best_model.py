import os
import json
import argparse

category = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science', 
            'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy', 
            'Economy', 'Math', 'Statistics', 'Chemistry']

llms = ['Moonshot']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="old", choices=['old', 'task2','task2_gen', 'task3'])
    args = parser.parse_args()

    task = args.task

    with open(f'{task}_retrain/best_hyperparams.json', 'r') as f:
        best_hyperparams = json.load(f) # distilbert

    with open(f'{task}_retrain_roberta/best_hyperparams.json', 'r') as f:
        best_hyperparams_roberta = json.load(f) # roberta

    best_final = {}

    for cat in category:
        best_final[cat] = {}
        for llm in llms:
            best_final[cat][llm] = {}
            if best_hyperparams[cat][llm][4] > best_hyperparams_roberta[cat][llm][4]:
                best_final[cat][llm] = best_hyperparams[cat][llm]
            else:
                best_final[cat][llm] = best_hyperparams_roberta[cat][llm]
    
    print(best_final)
    temp = f'{task}_final'
    if not os.path.exists(temp):
        os.makedirs(temp)
    with open(f'{temp}/best_hyperparams.json', 'w') as f:
        json.dump(best_final, f, indent=2)        
