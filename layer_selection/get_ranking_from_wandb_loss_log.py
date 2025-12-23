import json
from collections import defaultdict

FILE_TO_LOAD = "layer_selection_qwen_1_5b_training_loss_log.json"


try:
    with open(FILE_TO_LOAD, 'r') as f:
        all_runs_data = json.load(f)
    print(f"✅ Successfully loaded data from '{FILE_TO_LOAD}'.\n")
except FileNotFoundError:
    print(f"❌ Error: The file '{FILE_TO_LOAD}' was not found.")
    print("Please make sure the file is in the same directory as the script.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: The file '{FILE_TO_LOAD}' is not a valid JSON file.")
    exit()

data_by_step = defaultdict(list)


for run_name, run_data in all_runs_data.items():
    try:
        layer_str = run_name.split('/')[-2].split('_')[-1]
        layer_idx = int(layer_str)

        steps = run_data.get('train/global_step', [])
        losses = run_data.get('train/loss', [])

        for step, loss in zip(steps, losses):
            if step % 10 == 0:
                if loss is not None:
                    data_by_step[step].append({'layer': layer_idx, 'loss': loss})

    except (ValueError, IndexError):
        print(f"-> Warning: Could not parse layer index from run: '{run_name}'")

print("--- Layer Rankings by Training Loss (Best to Worst) ---")

sorted_steps = sorted(data_by_step.keys())

if not sorted_steps:
    print("\nNo data found for the pattern '/stage2' without '_reverse'.")
else:
    for step in sorted_steps:
        layer_performance = data_by_step[step]
        
        # Sort the list of dictionaries by the 'loss' value in ascending order
        # A lower loss is considered better.
        layer_performance_sorted = sorted(layer_performance, key=lambda x: x['loss'])
        
        best_to_worst_layers = [item['layer'] for item in layer_performance_sorted]
        
        print(f"\nStep {int(step)}:")
        print(best_to_worst_layers)
