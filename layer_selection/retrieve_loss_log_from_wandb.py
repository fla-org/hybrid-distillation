import wandb
import pandas as pd
import tqdm
import json

ENTITY = "YOUR_WANDB_ENTITY"
PROJECT = "YOUR_WANDB_PROJECT"
OUTPUT_FILE = "layer_selection_qwen_1_5b_training_loss_log.json"

all_runs_data = {}

api = wandb.Api()
project_path = f"{ENTITY}/{PROJECT}"


for layer_idx in tqdm.tqdm(range(28), desc="Processing Layers"):
    RUN_NAME = f"checkpoints/qwen2_1_5b_gdn_v4_hybrid_layer_selection_{layer_idx}/stage2"

    try:
        runs = api.runs(path=project_path, filters={"display_name": RUN_NAME})
        runs = [run for run in runs if run.state == "finished"]
        
        if runs:
            target_run = runs[0] 
            print(f"Found run: '{RUN_NAME}'")

            history = target_run.scan_history(keys=["train/global_step", "train/loss"])
            
            df = pd.DataFrame(history)

            if not df.empty and 'train/global_step' in df.columns and 'train/loss' in df.columns:
                
                df = df.dropna(subset=['train/global_step'])
                
                # Filter the DataFrame to keep only steps that are multiples of 500
                df_sampled = df[df['train/global_step'].astype(int) % 500 == 0]

                if not df_sampled.empty:
                    all_runs_data[RUN_NAME] = {
                        'train/global_step': df_sampled['train/global_step'].tolist(),
                        'train/loss': df_sampled['train/loss'].tolist()
                    }
                    print(f"-> Successfully processed and stored data for '{RUN_NAME}' (every 500th step).")
                else:
                    print(f"-> Warning: Run '{RUN_NAME}' has data, but no points at 500-step intervals.")
            else:
                print(f"-> Warning: Run '{RUN_NAME}' found, but it's empty or missing required keys.")

        else:
            print(f"-> Error: Run with name '{RUN_NAME}' not found.")

    except Exception as e:
        print(f"An error occurred while processing '{RUN_NAME}': {e}")

print("\nSaving all collected data to JSON file...")
try:
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_runs_data, f, indent=4)
    print(f"✅ Success! All data has been saved to '{OUTPUT_FILE}'")
except Exception as e:
    print(f"❌ Failed to save JSON file. Error: {e}")