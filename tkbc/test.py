# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import json
import os
import torch
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

from datasets import TemporalDataset
from models import ComplEx, TComplEx, TNTComplEx, TPComplExMetaFormer, TComplExMetaFormer, TNTComplExMetaFormer


def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_config(model_dir):
    """Load model configuration from saved directory"""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def create_model_from_config(config, sizes):
    """Create model instance from configuration"""
    model_name = config['model']
    rank = config['rank']
    no_time_emb = config.get('no_time_emb', False)
    
    if model_name == 'ComplEx':
        model = ComplEx(sizes, rank)
    elif model_name == 'TComplEx':
        model = TComplEx(sizes, rank, no_time_emb=no_time_emb)
    elif model_name == 'TNTComplEx':
        model = TNTComplEx(sizes, rank, no_time_emb=no_time_emb)
    elif model_name == 'TPComplExMetaFormer':
        model = TPComplExMetaFormer(
            sizes=sizes,
            rank=rank,
            num_metaformer_layers=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            no_time_emb=no_time_emb
        )
    elif model_name == 'TComplExMetaFormer':
        model = TComplExMetaFormer(
            sizes=sizes,
            rank=rank,
            num_metaformer_layers=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            no_time_emb=no_time_emb
        )
    elif model_name == 'TNTComplExMetaFormer':
        model = TNTComplExMetaFormer(
            sizes=sizes,
            rank=rank,
            num_metaformer_layers=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            no_time_emb=no_time_emb
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def load_model_weights(model, weights_path, device='cuda'):
    """Load model weights from checkpoint"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        model_class = checkpoint.get('model_class', 'unknown')
        return epoch, model_class
    else:
        # Assume it's a direct state dict
        model.load_state_dict(checkpoint)
        return 'unknown', 'unknown'


def load_individual_embeddings(weights_dir):
    """Load individual embedding files"""
    embeddings = {}
    
    # Entity embeddings
    ent_path = os.path.join(weights_dir, "entity_embeddings.pth")
    if os.path.exists(ent_path):
        embeddings['entity'] = torch.load(ent_path)
    
    # Relation embeddings
    rel_path = os.path.join(weights_dir, "relation_embeddings.pth")
    if os.path.exists(rel_path):
        embeddings['relation'] = torch.load(rel_path)
    
    # Time embeddings
    time_path = os.path.join(weights_dir, "time_embeddings.pth")
    if os.path.exists(time_path):
        embeddings['time'] = torch.load(time_path)
    
    # MetaFormer weights
    metaformer_path = os.path.join(weights_dir, "metaformer_weights.pth")
    if os.path.exists(metaformer_path):
        embeddings['metaformer'] = torch.load(metaformer_path)
    
    return embeddings


def find_available_epochs(model_dir):
    """Find all available epoch checkpoints"""
    weights_dir = os.path.join(model_dir, "weights")
    if not os.path.exists(weights_dir):
        return []
    
    epochs = []
    for item in os.listdir(weights_dir):
        if item.startswith("epoch_") and os.path.isdir(os.path.join(weights_dir, item)):
            try:
                epoch_num = int(item.split("_")[1])
                epochs.append(epoch_num)
            except ValueError:
                continue
    
    return sorted(epochs)


def evaluate_model(model, dataset, split='test', max_examples=-1):
    """Evaluate model on dataset"""
    model.eval()
    
    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """Aggregate metrics for missing lhs and rhs"""
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}
    
    with torch.no_grad():
        if dataset.has_intervals():
            results = dataset.eval(model, split, max_examples)
        else:
            results = avg_both(*dataset.eval(model, split, max_examples))
    
    return results


def run_prediction_example(model, dataset, num_examples=5):
    """Run prediction examples to show model capabilities"""
    model.eval()
    
    # Get some test examples
    test_data = dataset.get_test()
    if len(test_data) == 0:
        test_data = dataset.get_valid()
    
    if len(test_data) == 0:
        print("No test/validation data available for prediction examples")
        return
    
    # Sample random examples
    indices = np.random.choice(len(test_data), min(num_examples, len(test_data)), replace=False)
    examples = test_data[indices]
    
    print("\n" + "="*60)
    print("PREDICTION EXAMPLES")
    print("="*60)
    
    with torch.no_grad():
        for i, example in enumerate(examples):
            # Convert to tensor
            x = torch.from_numpy(example.reshape(1, -1)).cuda() if torch.cuda.is_available() else torch.from_numpy(example.reshape(1, -1))
            
            # Get score
            score = model.score(x)
            
            print(f"\nExample {i+1}:")
            print(f"  Query: (entity_{example[0]}, relation_{example[1]}, entity_{example[2]}, time_{example[3]})")
            print(f"  Score: {score.item():.4f}")
    
    print("="*60)


def analyze_embeddings(embeddings, logger):
    """Analyze loaded embeddings"""
    logger.info("EMBEDDING ANALYSIS")
    logger.info("="*50)
    
    for emb_type, emb_tensor in embeddings.items():
        if emb_type != 'metaformer' and isinstance(emb_tensor, torch.Tensor):
            shape = emb_tensor.shape
            mean_norm = torch.norm(emb_tensor, dim=1).mean().item()
            std_norm = torch.norm(emb_tensor, dim=1).std().item()
            
            logger.info(f"{emb_type.upper()} Embeddings:")
            logger.info(f"  Shape: {shape}")
            logger.info(f"  Mean L2 norm: {mean_norm:.4f}")
            logger.info(f"  Std L2 norm: {std_norm:.4f}")
            logger.info(f"  Min value: {emb_tensor.min().item():.4f}")
            logger.info(f"  Max value: {emb_tensor.max().item():.4f}")
            logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Load and test trained temporal models")
    
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help="Path to the saved model directory (e.g., ./runs/TComplEx_ICEWS14_...)"
    )
    
    parser.add_argument(
        '--epoch', type=int, default=None,
        help="Specific epoch to load (default: best model)"
    )
    
    parser.add_argument(
        '--split', type=str, choices=['train', 'valid', 'test'], default='test',
        help="Dataset split to evaluate on"
    )
    
    parser.add_argument(
        '--max_examples', type=int, default=-1,
        help="Maximum number of examples to evaluate (-1 for all)"
    )
    
    parser.add_argument(
        '--show_predictions', action='store_true',
        help="Show prediction examples"
    )
    
    parser.add_argument(
        '--analyze_embeddings', action='store_true',
        help="Analyze embedding statistics"
    )
    
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to use for inference"
    )

    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        return
    
    logger.info("="*80)
    logger.info("TEMPORAL MODEL LOADER AND TESTER")
    logger.info("="*80)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Load configuration
        logger.info("Loading model configuration...")
        config = load_model_config(args.model_dir)
        
        logger.info(f"Model: {config['model']}")
        logger.info(f"Dataset: {config['dataset']}")
        logger.info(f"Rank: {config['rank']}")
        logger.info(f"Training epochs: {config.get('max_epochs', 'unknown')}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = TemporalDataset(config['dataset'])
        sizes = dataset.get_shape()
        
        logger.info(f"Dataset shape: {sizes}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model_from_config(config, sizes)
        model = model.to(args.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        # Determine which epoch to load
        if args.epoch is not None:
            # Load specific epoch
            epoch_to_load = args.epoch
            weights_dir = os.path.join(args.model_dir, "weights", f"epoch_{epoch_to_load:03d}")
        else:
            # Load best model
            best_model_dir = os.path.join(args.model_dir, "best_model")
            if os.path.exists(best_model_dir):
                weights_dir = best_model_dir
                # Try to extract epoch number
                if os.path.islink(best_model_dir):
                    target = os.readlink(best_model_dir)
                    try:
                        epoch_to_load = int(os.path.basename(target).split("_")[1])
                    except:
                        epoch_to_load = "best"
                else:
                    epoch_to_load = "best"
            else:
                # Find latest epoch
                available_epochs = find_available_epochs(args.model_dir)
                if not available_epochs:
                    logger.error("No saved epochs found!")
                    return
                epoch_to_load = available_epochs[-1]
                weights_dir = os.path.join(args.model_dir, "weights", f"epoch_{epoch_to_load:03d}")
        
        logger.info(f"Loading weights from epoch: {epoch_to_load}")
        
        # Load model weights
        model_path = os.path.join(weights_dir, "model.pth")
        if os.path.exists(model_path):
            logger.info("Loading complete model weights...")
            epoch, model_class = load_model_weights(model, model_path, args.device)
            logger.info(f"Loaded model from epoch {epoch} (class: {model_class})")
        else:
            logger.warning("Complete model file not found, trying individual embeddings...")
            # Load individual embeddings (fallback)
            embeddings = load_individual_embeddings(weights_dir)
            if embeddings:
                logger.info("Loading individual embedding components...")
                if 'entity' in embeddings:
                    model.embeddings[0].weight.data = embeddings['entity'].to(args.device)
                if 'relation' in embeddings:
                    model.embeddings[1].weight.data = embeddings['relation'].to(args.device)
                if 'time' in embeddings and len(model.embeddings) > 2:
                    model.embeddings[2].weight.data = embeddings['time'].to(args.device)
                logger.info("Individual embeddings loaded successfully")
            else:
                logger.error("No model weights found!")
                return
        
        # Analyze embeddings if requested
        if args.analyze_embeddings:
            embeddings = load_individual_embeddings(weights_dir)
            if embeddings:
                analyze_embeddings(embeddings, logger)
        
        # Evaluate model
        logger.info(f"Evaluating model on {args.split} split...")
        eval_start_time = datetime.now()
        
        results = evaluate_model(model, dataset, args.split, args.max_examples)
        
        eval_time = (datetime.now() - eval_start_time).total_seconds()
        logger.info(f"Evaluation completed in {eval_time:.2f}s")
        
        # Log results
        logger.info("="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        
        if isinstance(results, dict):
            if 'MRR' in results:
                mrr = results['MRR']
                hits = results.get('hits@[1,3,10]', [0, 0, 0])
                logger.info(f"MRR: {mrr:.4f}")
                logger.info(f"Hits@1: {hits[0]:.4f}")
                logger.info(f"Hits@3: {hits[1]:.4f}")
                logger.info(f"Hits@10: {hits[2]:.4f}")
            else:
                logger.info(f"Results: {results}")
        else:
            logger.info(f"Results: {results}")
        
        # Show prediction examples if requested
        if args.show_predictions:
            run_prediction_example(model, dataset)
        
        # Save evaluation results
        eval_results = {
            'model_dir': args.model_dir,
            'epoch': epoch_to_load,
            'split': args.split,
            'max_examples': args.max_examples,
            'results': results if isinstance(results, dict) else str(results),
            'evaluation_time': eval_time,
            'timestamp': datetime.now().isoformat()
        }
        
        eval_file = os.path.join(args.model_dir, f"eval_{args.split}_epoch_{epoch_to_load}.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to: {eval_file}")
        
    except Exception as e:
        logger.error(f"Error during model loading/testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()