import pathlib
import torch
import ankh
import numpy as np
import pandas as pd
import time
import argparse
from Bio import SeqIO
from tqdm import tqdm
from engine import ProteinLanguageModel  # Assuming the previous class is in this file

def process_sequences(model_name, fasta_path, output_dir, batch_size=100):
    """
    Process sequences using specified PLM and save embeddings
    
    Args:
        model_name (str): Name of the PLM to use
        fasta_path (str): Path to input FASTA file
        output_dir (str): Directory to save outputs
        batch_size (int): Number of sequences to process before saving
    """
    # Initialize model
    plm = ProteinLanguageModel(
        model_name=model_name,
        max_length=1200,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create output directory if it doesn't exist
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize lists for storing results
    ids_list = []
    embed_vects_list = []
    
    # Start timing
    t0 = time.time()
    checkpoint = 0
    count = 0
    
    print(f"Processing sequences with {model_name}...")
    fasta_sequences = SeqIO.parse(fasta_path, "fasta")
    
    for item in tqdm(fasta_sequences):
        if count>=2:
          break
        else:
          count+=1
          ids_list.append(item.id)
          # Get embeddings for sequence
          embeddings = plm.get_embeddings(
              sequence=str(item.seq),
              pool_method='mean'
          )
          embed_vects_list.append(embeddings.cpu().numpy())
          
          checkpoint += 1
          if checkpoint >= batch_size:
              # Save intermediate results
              np.save(
                  output_dir / f'{model_name}_ids_checkpoint.npy',
                  np.array(ids_list)
              )
              np.save(
                  output_dir / f'{model_name}_embeddings_checkpoint.npy',
                  np.array(embed_vects_list)
              )
              checkpoint = 0
    
    # Save final results
    np.save(
        output_dir / f'{model_name}_ids.npy',
        np.array(ids_list)
    )
    np.save(
        output_dir / f'{model_name}_embeddings.npy',
        np.array(embed_vects_list)
    )
    
    print(f'Total elapsed time for {model_name}: {time.time()-t0:.2f} seconds')

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process FASTA sequences with different models')
    
    # Add arguments
    parser.add_argument('--fasta_path', type=str, default="./dataset/data.fasta",
                        help='Path to input FASTA file')
    parser.add_argument('--output_dir', type=str, default="embeddings_output",
                        help='Directory for output files (default: embeddings_output)')
    parser.add_argument('--models', nargs='+', default=['esm2', 'bert', 't5', 'ankh'],
                        help='List of models to use (default: esm2 bert t5 ankh)')
    
    # Parse arguments
    args = parser.parse_args()
    # Set paths
    fasta_path = args.fasta_path
    output_dir = args.output_dir
    
    # List of models to process
    models = ['esm2', 'bert', 't5', 'ankh']
    
    # Process sequences with each model
    for model_name in models:
        try:
            process_sequences(
                model_name=model_name,
                fasta_path=fasta_path,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
'''
 run command 
 python script.py --fasta_path /path/to/data.fasta --output_dir my_outputs
'''