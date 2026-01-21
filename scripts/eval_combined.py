#!/usr/bin/env python3

import os
import sys
import re
import pandas as pd
from pathlib import Path

# Trackastra imports
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc
from trackastra.data import load_tiff_timeseries

# Traccuracy imports
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics
from traccuracy import run_metrics

def extract_epoch_from_checkpoint(checkpoint_name):
    """Extract epoch number from checkpoint name like '2025-09-21_20-19-56_example_14'"""
    match = re.search(r'_(\d+)$', checkpoint_name)
    if match:
        return int(match.group(1))
    return None

def find_checkpoints(base_name):
    """Find all checkpoint directories matching the base name pattern"""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Error: runs directory not found")
        return []
    
    pattern = f"{base_name}_*"
    checkpoints = []
    
    for path in runs_dir.glob(pattern):
        if path.is_dir():
            checkpoint_name = path.name
            epoch = extract_epoch_from_checkpoint(checkpoint_name)
            if epoch is not None:
                checkpoints.append((epoch, checkpoint_name))
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def evaluate_checkpoint(checkpoint_name, device="cpu"):
    """Evaluate a single checkpoint and return TRA, DET, LNK metrics"""
    
    print(f"Loading model from runs/{checkpoint_name}...")
    
    if True:
        # Load model
        model = Trackastra.from_folder(Path(f'runs/{checkpoint_name}'), device=device)
        
        print(model.transformer.config)

        # Load masks for ground truth
        masks = load_tiff_timeseries(Path('data/ctc/Fluo-N2DL-HeLa/02_GT/TRA/'))
        print(f"Masks loaded: {masks.shape}")
        
        # Perform tracking
        print("Running tracking...")
        maester_embeddings_path = 'data/ctc/Fluo-N2DL-HeLa/02.pt'
        track_graph, masks = model.track_from_disk(
            Path('data/ctc/Fluo-N2DL-HeLa/02/'), 
            Path('data/ctc/Fluo-N2DL-HeLa/02_GT/TRA/'), 
            mode="greedy", 
            maester_embeddings_path=maester_embeddings_path
        )
        
        # Create output directory
        output_dir = f"models_ctc_tracks/{checkpoint_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to CTC format
        print("Converting to CTC format...")
        ctc_tracks, masks_tracked = graph_to_ctc(
            track_graph,
            masks,
            outdir=output_dir,
        )
        
        # Load ground truth data
        print("Loading ground truth data...")
        gt_data = load_ctc_data(
            "data/ctc/Fluo-N2DL-HeLa/02_GT/TRA",
            "data/ctc/Fluo-N2DL-HeLa/02_GT/TRA/man_track.txt",
            name="GT"
        )
        
        # Load predicted data
        print("Loading predicted data...")
        pred_data = load_ctc_data(
            output_dir,
            f"{output_dir}/man_track.txt",
            name="prediction"
        )
        
        # Run metrics
        print("Computing metrics...")
        results, matched = run_metrics(
            gt_data=gt_data,
            pred_data=pred_data,
            matcher=CTCMatcher(),
            metrics=[CTCMetrics()],
        )
        
        # Extract metrics
        if results and len(results) > 0:
            metrics = results[0]['results']
            return {
                'TRA': metrics.get('TRA', None),
                'DET': metrics.get('DET', None), 
                'LNK': metrics.get('LNK', None)
            }
        else:
            print("No results returned from metrics computation")
            return None
            
    else:
        #print(f"Error evaluating checkpoint {checkpoint_name}: {e}")
        print("Error")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_full.py <base_name>")
        print("Example: python eval_full.py 2025-09-21_20-19-56_example")
        sys.exit(1)
    
    base_name = sys.argv[1]
    device = "cuda"  # Can be changed to "cuda" or "mps" if available
    
    # Find all matching checkpoints
    checkpoints = find_checkpoints(base_name)
    
    if not checkpoints:
        print(f"No checkpoints found matching pattern: {base_name}_*")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for epoch, checkpoint_name in checkpoints:
        print(f"  Epoch {epoch}: {checkpoint_name}")
    
    # Create output directory
    os.makedirs("models_ctc_tracks", exist_ok=True)
    
    # Process each checkpoint
    results = []
    
    for epoch, checkpoint_name in checkpoints:
        print(f"\n{'='*60}")
        print(f"Processing epoch {epoch}: {checkpoint_name}")
        print(f"{'='*60}")
        
        metrics = evaluate_checkpoint(checkpoint_name, device)
        
        if metrics is not None:
            results.append({
                'epoch': epoch,
                'TRA': metrics['TRA'],
                'DET': metrics['DET'],
                'LNK': metrics['LNK']
            })
            print(f"Results for epoch {epoch}:")
            print(f"  TRA: {metrics['TRA']:.6f}")
            print(f"  DET: {metrics['DET']:.6f}")
            print(f"  LNK: {metrics['LNK']:.6f}")
        else:
            print(f"Failed to get metrics for epoch {epoch}")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = f"{base_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")
        print(df.to_string(index=False, float_format='%.6f'))
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"Best TRA: {df['TRA'].max():.6f} (epoch {df.loc[df['TRA'].idxmax(), 'epoch']})")
        print(f"Best DET: {df['DET'].max():.6f} (epoch {df.loc[df['DET'].idxmax(), 'epoch']})")
        print(f"Best LNK: {df['LNK'].max():.6f} (epoch {df.loc[df['LNK'].idxmax(), 'epoch']})")
    else:
        print("No results to save")

if __name__ == "__main__":
    main()
