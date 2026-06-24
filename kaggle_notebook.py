"""
ColorWars — Kaggle Notebook for MCTS Self-Play + Neural Network Training
=========================================================================

Upload the colorwars_kaggle.zip as a Kaggle Dataset, then run this notebook.

This notebook:
  1. Extracts the project files
  2. Generates MCTS self-play training data (50 games, 200 sims/move)
  3. Trains the AlphaZero-style dual-head CNN
  4. Saves everything to /kaggle/working/ for download

Expected runtime: ~30-60 min on Kaggle CPU (faster with GPU)
"""

# ═══════════════════════════════════════════════════════════════════
#  Cell 1: Setup — Extract project files
# ═══════════════════════════════════════════════════════════════════

import os
import sys
import zipfile
import shutil

# Find the uploaded zip (Kaggle datasets go to /kaggle/input/)
INPUT_DIR = "/kaggle/input"
OUTPUT_DIR = "/kaggle/working"
PROJECT_DIR = os.path.join(OUTPUT_DIR, "ColorWars")

# Look for the zip file in input directories
zip_path = None
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.endswith('.zip') and 'colorwars' in f.lower():
            zip_path = os.path.join(root, f)
            break
    if zip_path:
        break

# Fallback: look for any zip
if zip_path is None:
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith('.zip'):
                zip_path = os.path.join(root, f)
                break
        if zip_path:
            break

if zip_path:
    print(f"📦 Found zip: {zip_path}")
    # Extract
    if os.path.exists(PROJECT_DIR):
        shutil.rmtree(PROJECT_DIR)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(PROJECT_DIR)
    print(f"✅ Extracted to: {PROJECT_DIR}")
else:
    # Maybe files are already extracted (uploaded as dataset folder)
    # Look for the ai/ directory
    for root, dirs, files in os.walk(INPUT_DIR):
        if 'ai' in dirs:
            # Copy the project
            if os.path.exists(PROJECT_DIR):
                shutil.rmtree(PROJECT_DIR)
            shutil.copytree(root, PROJECT_DIR)
            print(f"✅ Copied project from: {root}")
            break
    else:
        print("❌ Could not find ColorWars project files!")
        print(f"   Contents of {INPUT_DIR}:")
        for root, dirs, files in os.walk(INPUT_DIR):
            for f in files:
                print(f"   {os.path.join(root, f)}")
        raise FileNotFoundError("Upload colorwars_kaggle.zip as a Kaggle Dataset")

# Add to Python path
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

# Verify
print(f"\n📂 Project structure:")
for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, "ai")):
    # Skip __pycache__
    dirs[:] = [d for d in dirs if d != '__pycache__']
    level = root.replace(PROJECT_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for f in sorted(files):
        if not f.endswith('.pyc'):
            print(f"{subindent}{f}")

# ═══════════════════════════════════════════════════════════════════
#  Cell 2: Generate MCTS Self-Play Data
# ═══════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  PHASE 1: GENERATING MCTS SELF-PLAY DATA")
print("═" * 60)

from ai.self_play.training_pipeline import TrainingPipeline

# ── Configuration ──
# Adjust these based on how much time you have:
#   Quick test:  games=10,  sims=50   (~5 min)
#   Standard:    games=50,  sims=100  (~30 min)
#   Full:        games=100, sims=200  (~2 hrs)
N_GAMES = 50
N_SIMULATIONS = 200
TEMPERATURE = 1.0
TEMP_THRESHOLD = 8
SEED = 42
SAVE_DIR = os.path.join(PROJECT_DIR, "data", "mcts_self_play")

print(f"\n  Config:")
print(f"    Games:          {N_GAMES}")
print(f"    Sims/move:      {N_SIMULATIONS}")
print(f"    Temperature:    {TEMPERATURE} → greedy after move {TEMP_THRESHOLD}")
print(f"    Save dir:       {SAVE_DIR}")
print()

pipeline = TrainingPipeline(grid_size=6)

records = pipeline.generate(
    n_games=N_GAMES,
    n_simulations=N_SIMULATIONS,
    temperature=TEMPERATURE,
    temperature_threshold=TEMP_THRESHOLD,
    save_dir=SAVE_DIR,
    seed=SEED,
    swap_colors=True,
    verbose=True,  # 👈 Added this so you can see the simulation live
)

# Show data statistics
examples = pipeline.extract_examples(records, augment=False)
stats = pipeline.get_data_stats(examples)
pipeline.print_data_stats(stats)

# Show augmented count
aug_examples = pipeline.extract_examples(records, augment=True)
aug_stats = pipeline.get_data_stats(aug_examples)
print(f"  With 8× augmentation:")
print(f"    Raw examples:       {stats['total_examples']}")
print(f"    Augmented examples: {aug_stats['total_examples']}")
print(f"    Multiplier:         "
      f"{aug_stats['total_examples']/max(1,stats['total_examples']):.1f}×")

# Verify saved files
saved_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.json')]
print(f"\n  📁 Saved {len(saved_files)} game files to: {SAVE_DIR}/")

# ═══════════════════════════════════════════════════════════════════
#  Cell 3: Train the Neural Network
# ═══════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  PHASE 2: TRAINING NEURAL NETWORK")
print("═" * 60)

import torch
from ai.neural_net.model import ColorWarsNet, get_device
from ai.neural_net.trainer import Trainer
from ai.neural_net import config

device = get_device()
print(f"\n  PyTorch: {torch.__version__}")
print(f"  Device:  {device}")
if torch.cuda.is_available():
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")

# Create model
model = ColorWarsNet()
model.to(device)
model.summary()

# Create trainer
trainer = Trainer(
    model=model,
    device=device,
    lr=0.001,
    weight_decay=1e-4,
    batch_size=64,
)

# Load MCTS self-play data
n_train, n_val = trainer.load_data(
    data_dir=SAVE_DIR,
    augment=True,       # 8× symmetry augmentation
    policy_only=True,   # Only MCTS examples
    perspective=True,   # Canonical (current player's POV)
)

# Train
metrics = trainer.train(n_epochs=20)

# Save checkpoint
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "ai", "neural_net", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "colorwars_net_v1.pt")
model.save_checkpoint(
    checkpoint_path,
    epoch=20,
    optimizer_state=trainer.optimizer.state_dict(),
    metadata={'training_history': metrics},
)
print(f"\n  💾 Checkpoint saved: {checkpoint_path}")

# ═══════════════════════════════════════════════════════════════════
#  Cell 4: Evaluate — NN Agent vs Random Agent
# ═══════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  PHASE 3: EVALUATION — NN Agent vs Random Agent")
print("═" * 60)

from ai.agents.nn_agent import NNAgent
from ai.agents.random_agent import RandomAgent
from ai.self_play.self_play_manager import SelfPlayManager

nn_agent = NNAgent(model=model, name="NeuralNet", temperature=0.01, seed=42)
random_agent = RandomAgent(name="Random", seed=99)

manager = SelfPlayManager(grid_size=6)
eval_records = manager.run_self_play(
    agent1=nn_agent,
    agent2=random_agent,
    n_games=20,
    verbose=False,
)

eval_stats = manager.get_statistics(eval_records)
manager.print_statistics(eval_stats)

# ═══════════════════════════════════════════════════════════════════
#  Cell 5: Package outputs for download
# ═══════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("  PACKAGING OUTPUTS FOR DOWNLOAD")
print("═" * 60)

# Create a zip of the outputs
output_zip = os.path.join(OUTPUT_DIR, "colorwars_trained.zip")
with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add checkpoint
    zf.write(checkpoint_path, "checkpoints/colorwars_net_v1.pt")

    # Add MCTS game data
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.endswith('.json'):
            zf.write(
                os.path.join(SAVE_DIR, f),
                f"mcts_self_play/{f}",
            )

print(f"\n  📦 Output zip: {output_zip}")
print(f"     Contents:")
with zipfile.ZipFile(output_zip, 'r') as zf:
    for info in zf.infolist():
        print(f"       {info.filename} ({info.file_size:,} bytes)")

total_size = os.path.getsize(output_zip)
print(f"\n  Total size: {total_size / 1024 / 1024:.1f} MB")
print(f"\n  ✅ Download 'colorwars_trained.zip' from the Output tab!")
print(f"     Then extract into your ColorWars project:")
print(f"       - checkpoints/ → ai/neural_net/checkpoints/")
print(f"       - mcts_self_play/ → data/mcts_self_play/")

print("\n" + "═" * 60)
print("  ALL DONE! 🎉")
print("═" * 60)
