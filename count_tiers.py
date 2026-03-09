#!/usr/bin/env python3
import json
import os

# Key experiment results
files = [
    'experiments/results_level2_pamap2_adaptive_auto_hz.json',
    'experiments/results_all_levels_pamap2_auto_hz.json', 
    'experiments/results_ptt_ppg_final.json',
    'experiments/results_level4_pamap2.json'
]

total = {'GOLD': 0, 'SILVER': 0, 'BRONZE': 0, 'REJECTED': 0}

for file in files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        
        print(f'=== {file} ===')
        
        # Find tier_counts
        def find_counts(obj, name=''):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'tier_counts' and isinstance(v, dict):
                        print(f'{name}: {v}')
                        for tier, count in v.items():
                            if tier in total:
                                total[tier] += count
                    else:
                        find_counts(v, f'{name}.{k}' if name else k)
        
        find_counts(data)
        print()
        
    except Exception as e:
        print(f'Error: {e}')

print('=== TOTAL ACROSS KEY EXPERIMENTS ===')
print(f'GOLD: {total["GOLD"]}')
print(f'SILVER: {total["SILVER"]}')
print(f'BRONZE: {total["BRONZE"]}')
print(f'REJECTED: {total["REJECTED"]}')
total_wins = sum(total.values())
print(f'Total windows: {total_wins}')
print(f'GOLD+REJECTED (certain): {total["GOLD"] + total["REJECTED"]} ({(total["GOLD"] + total["REJECTED"])/total_wins*100:.1f}%)')
print(f'SILVER+BRONZE (ambiguous): {total["SILVER"] + total["BRONZE"]} ({(total["SILVER"] + total["BRONZE"])/total_wins*100:.1f}%)')
