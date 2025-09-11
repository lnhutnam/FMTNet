import argparse
from typing import Dict, List, Tuple, Any
import logging
import torch
from torch import optim
import os
import json
import time
from datetime import datetime
import numpy as np
import itertools
from collections import defaultdict

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, TPComplExMetaFormer, TComplExMetaFormer, TNTComplExMetaFormer, TPComplEx
from regularizers import N3, Lambda3


def setup_logging_and_directories(args, hyperparams=None):
    """Setup logging and create necessary directories"""
    
    # Create model_id based on parameters
    if hyperparams:
        model_id = f"{args.model}_{args.dataset}_rank{hyperparams['rank']}_" \
                   f"layers{hyperparams.get('metaformer_layers', 'na')}_" \
                   f"mlp{hyperparams.get('mlp_ratio', 'na')}_" \
                   f"drop{hyperparams.get('drop_path_rate', 'na')}_" \
                   f"lr{args.learning_rate}_embreg{args.emb_reg}_timereg{args.time_reg}_" \
                   f"bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        model_id = f"{args.model}_{args.dataset}_rank{args.rank}_lr{args.learning_rate}_" \
                   f"embreg{args.emb_reg}_timereg{args.time_reg}_bs{args.batch_size}_" \
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create runs directory structure
    if args.grid_search:
        base_runs_dir = "./runs/grid_search"
        search_session_id = f"search_{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(base_runs_dir, search_session_id)
        model_dir = os.path.join(session_dir, model_id)
    else:
        runs_dir = "./runs"
        model_dir = os.path.join(runs_dir, model_id)
        session_dir = None
    
    logs_dir = os.path.join(model_dir, "logs")
    weights_dir = os.path.join(model_dir, "weights")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(logs_dir, "training.log")
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Log experiment configuration
    logger.info("="*80)
    logger.info("TEMPORAL KNOWLEDGE GRAPH EMBEDDING TRAINING")
    if args.grid_search:
        logger.info("GRID SEARCH MODE")
    logger.info("="*80)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Experiment directory: {model_dir}")
    if session_dir:
        logger.info(f"Search session directory: {session_dir}")
    logger.info("="*80)
    
    # Save configuration
    config = vars(args).copy()
    config['model_id'] = model_id
    config['start_time'] = datetime.now().isoformat()
    if hyperparams:
        config.update(hyperparams)
    
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Configuration saved to: " + config_file)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return model_id, model_dir, logs_dir, weights_dir, logger, session_dir


def save_model_weights(model, epoch, weights_dir, logger, is_best=False):
    """Save model weights including embeddings"""
    
    # Create epoch-specific directory
    epoch_dir = os.path.join(weights_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    try:
        # Save complete model state
        model_path = os.path.join(epoch_dir, "model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
        }, model_path)
        
        # Save individual embedding components
        embeddings_info = {}
        
        # Entity embeddings
        if hasattr(model, 'embeddings') and len(model.embeddings) > 0:
            ent_emb_path = os.path.join(epoch_dir, "entity_embeddings.pth")
            torch.save(model.embeddings[0].weight.data, ent_emb_path)
            embeddings_info['entity_embeddings'] = {
                'path': ent_emb_path,
                'shape': list(model.embeddings[0].weight.shape),
                'num_parameters': model.embeddings[0].weight.numel()
            }
            
            # Relation embeddings
            if len(model.embeddings) > 1:
                rel_emb_path = os.path.join(epoch_dir, "relation_embeddings.pth")
                torch.save(model.embeddings[1].weight.data, rel_emb_path)
                embeddings_info['relation_embeddings'] = {
                    'path': rel_emb_path,
                    'shape': list(model.embeddings[1].weight.shape),
                    'num_parameters': model.embeddings[1].weight.numel()
                }
            
            # Time embeddings (if exists)
            if len(model.embeddings) > 2:
                time_emb_path = os.path.join(epoch_dir, "time_embeddings.pth")
                torch.save(model.embeddings[2].weight.data, time_emb_path)
                embeddings_info['time_embeddings'] = {
                    'path': time_emb_path,
                    'shape': list(model.embeddings[2].weight.shape),
                    'num_parameters': model.embeddings[2].weight.numel()
                }
        
        # Save embedding info
        emb_info_path = os.path.join(epoch_dir, "embeddings_info.json")
        with open(emb_info_path, 'w') as f:
            json.dump(embeddings_info, f, indent=2)
        
        # Save MetaFormer weights if applicable
        if hasattr(model, 'metaformer_layers') and len(model.metaformer_layers) > 0:
            metaformer_path = os.path.join(epoch_dir, "metaformer_weights.pth")
            metaformer_state = {}
            for i, layer in enumerate(model.metaformer_layers):
                metaformer_state[f'layer_{i}'] = layer.state_dict()
            torch.save(metaformer_state, metaformer_path)
            
            embeddings_info['metaformer_layers'] = {
                'path': metaformer_path,
                'num_layers': len(model.metaformer_layers),
                'layer_dim': getattr(model.metaformer_layers[0], 'norm1', None).weight.shape[0] if hasattr(model.metaformer_layers[0], 'norm1') else 'unknown'
            }
        
        logger.info(f"Model weights saved to: {epoch_dir}")
        logger.info(f"  - Complete model: {model_path}")
        logger.info(f"  - Embeddings info: {emb_info_path}")
        
        for emb_type, info in embeddings_info.items():
            if 'shape' in info:
                logger.info(f"  - {emb_type}: {info['path']} (shape: {info['shape']}, params: {info['num_parameters']:,})")
        
        # Create best model symlink if this is the best epoch
        if is_best:
            best_dir = os.path.join(os.path.dirname(weights_dir), "best_model")
            if os.path.exists(best_dir):
                os.remove(best_dir) if os.path.isfile(best_dir) else os.rmdir(best_dir)
            os.symlink(epoch_dir, best_dir)
            logger.info(f"Best model symlink created: {best_dir}")
            
    except Exception as e:
        logger.error(f"Error saving model weights: {str(e)}")


def log_metrics(metrics, split, epoch, logger, metrics_file):
    """Log metrics and save to file"""
    
    # Log to console and file
    logger.info(f"Epoch {epoch:3d} - {split.upper()} Metrics:")
    
    if isinstance(metrics, dict):
        if 'MRR' in metrics:
            # Standard format
            mrr = metrics['MRR']
            hits = metrics.get('hits@[1,3,10]', [0, 0, 0])
            logger.info(f"  MRR: {mrr:.4f}")
            logger.info(f"  Hits@1: {hits[0]:.4f}")
            logger.info(f"  Hits@3: {hits[1]:.4f}")
            logger.info(f"  Hits@10: {hits[2]:.4f}")
            
            # Save to metrics file
            metrics_data = {
                'epoch': epoch,
                'split': split,
                'mrr': float(mrr),
                'hits_1': float(hits[0]),
                'hits_3': float(hits[1]),
                'hits_10': float(hits[2]),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Handle other metric formats
            logger.info(f"  Metrics: {metrics}")
            metrics_data = {
                'epoch': epoch,
                'split': split,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
    else:
        # Handle non-dict metrics
        logger.info(f"  Value: {metrics}")
        metrics_data = {
            'epoch': epoch,
            'split': split,
            'value': float(metrics) if isinstance(metrics, (int, float, np.number)) else str(metrics),
            'timestamp': datetime.now().isoformat()
        }
    
    # Append to metrics file
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics_data) + '\n')
    
    return metrics_data


def track_training_progress(epoch, start_time, max_epochs, logger):
    """Track and log training progress"""
    elapsed_time = time.time() - start_time
    progress = (epoch + 1) / max_epochs
    estimated_total = elapsed_time / progress if progress > 0 else 0
    eta = estimated_total - elapsed_time
    
    logger.info(f"Training Progress:")
    logger.info(f"  Epoch: {epoch + 1}/{max_epochs} ({progress*100:.1f}%)")
    logger.info(f"  Elapsed: {elapsed_time/3600:.2f}h")
    logger.info(f"  ETA: {eta/3600:.2f}h")
    logger.info(f"  Total estimated: {estimated_total/3600:.2f}h")


def generate_hyperparameter_combinations(args) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters for grid search"""
    
    # Define hyperparameter grids
    hyperparams = {
        'rank': args.rank_grid,
        'metaformer_layers': args.metaformer_layers_grid,
        'mlp_ratio': args.mlp_ratio_grid,
        'drop_path_rate': args.drop_path_rate_grid
    }
    
    # Filter hyperparameters based on model type
    if args.model in ['ComplEx', 'TComplEx', 'TNTComplEx']:
        # Non-MetaFormer models don't use these parameters
        combinations = [{'rank': rank} for rank in args.rank_grid]
    else:
        # MetaFormer models use all parameters
        keys = list(hyperparams.keys())
        values = list(hyperparams.values())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    return combinations


def save_grid_search_results(session_dir, results, logger):
    """Save comprehensive grid search results"""
    
    # Sort results by validation MRR (descending)
    sorted_results = sorted(results, key=lambda x: x.get('best_valid_mrr', -1), reverse=True)
    
    # Save detailed results
    results_file = os.path.join(session_dir, "grid_search_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'search_completed': datetime.now().isoformat(),
            'total_combinations': len(results),
            'best_combination': sorted_results[0] if sorted_results else None,
            'all_results': sorted_results
        }, f, indent=2)
    
    # Save summary table
    summary_file = os.path.join(session_dir, "grid_search_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("GRID SEARCH RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total combinations tested: {len(results)}\n")
        f.write(f"Search completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TOP 10 CONFIGURATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Layers':<7} {'MLP':<5} {'Drop':<6} {'MRR':<8} {'Model ID':<40}\n")
        f.write("-" * 80 + "\n")
        
        for i, result in enumerate(sorted_results[:10]):
            hyperparams = result['hyperparameters']
            f.write(f"{hyperparams.get('rank', 'N/A'):<6} "
                   f"{hyperparams.get('metaformer_layers', 'N/A'):<7} "
                   f"{hyperparams.get('mlp_ratio', 'N/A'):<5} "
                   f"{hyperparams.get('drop_path_rate', 'N/A'):<6.1f} "
                   f"{result.get('best_valid_mrr', -1):<8.4f} "
                   f"{result['model_id']}\n")
        
        # Statistics
        if results:
            mrrs = [r.get('best_valid_mrr', -1) for r in results if r.get('best_valid_mrr', -1) > 0]
            if mrrs:
                f.write(f"\nSTATISTICS:\n")
                f.write(f"Best MRR: {max(mrrs):.4f}\n")
                f.write(f"Mean MRR: {np.mean(mrrs):.4f}\n")
                f.write(f"Std MRR: {np.std(mrrs):.4f}\n")
                f.write(f"Median MRR: {np.median(mrrs):.4f}\n")
    
    logger.info(f"Grid search results saved to: {results_file}")
    logger.info(f"Grid search summary saved to: {summary_file}")
    
    return sorted_results[0] if sorted_results else None


def train_single_configuration(args, hyperparams, dataset, combination_idx, total_combinations):
    """Train a single hyperparameter configuration"""
    
    # Setup directories and logging for this configuration
    model_id, model_dir, logs_dir, weights_dir, logger, session_dir = setup_logging_and_directories(args, hyperparams)
    
    logger.info(f"GRID SEARCH: Configuration {combination_idx + 1}/{total_combinations}")
    logger.info(f"Hyperparameters: {hyperparams}")
    
    # Initialize metrics file
    metrics_file = os.path.join(logs_dir, "metrics.jsonl")
    
    # Get dataset sizes
    sizes = dataset.get_shape()
    
    # Initialize model with current hyperparameters
    logger.info(f"Initializing model: {args.model}")
    
    if args.model == 'ComplEx':
        model = ComplEx(sizes, hyperparams['rank'])
    elif args.model == 'TComplEx':
        model = TComplEx(sizes, hyperparams['rank'], no_time_emb=args.no_time_emb)
    elif args.model == 'TNTComplEx':
        model = TNTComplEx(sizes, hyperparams['rank'], no_time_emb=args.no_time_emb)
    elif args.model == 'TPComplEx':
        model = TPComplEx(sizes, hyperparams['rank'], no_time_emb=args.no_time_emb)
    elif args.model == 'TPComplExMetaFormer':
        model = TPComplExMetaFormer(
            sizes=sizes, 
            rank=hyperparams['rank'],
            no_time_emb=args.no_time_emb
        )
    elif args.model == 'TComplExMetaFormer':
        model = TComplExMetaFormer(
            sizes=sizes,
            rank=hyperparams['rank'], 
            no_time_emb=args.no_time_emb
        )
    elif args.model == 'TNTComplExMetaFormer':
        model = TNTComplExMetaFormer(
            sizes=sizes,
            rank=hyperparams['rank'],
            no_time_emb=args.no_time_emb
        )
    
    model = model.cuda()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model initialized:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    # Initialize optimizer and regularizers
    opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    emb_reg = N3(args.emb_reg)
    time_reg = Lambda3(args.time_reg)
    
    # Training variables
    start_time = time.time()
    best_valid_mrr = -1.0
    best_epoch = -1
    
    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """Aggregate metrics for missing lhs and rhs"""
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}
    
    logger.info("Starting training for this configuration...")
    
    # Training loop
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        
        # Get training examples
        examples = torch.from_numpy(dataset.get_train().astype('int64'))
        
        # Training
        model.train()
        if dataset.has_intervals():
            optimizer = IKBCOptimizer(
                model, emb_reg, time_reg, opt, dataset,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)
        else:
            optimizer = TKBCOptimizer(
                model, emb_reg, time_reg, opt,
                batch_size=args.batch_size
            )
            optimizer.epoch(examples)
        
        epoch_training_time = time.time() - epoch_start_time
        
        # Validation and testing
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            logger.info(f"Evaluating model at epoch {epoch + 1}...")
            eval_start_time = time.time()
            
            if dataset.has_intervals():
                valid, test, train = [
                    dataset.eval(model, split, -1 if split != 'train' else 50000)
                    for split in ['valid', 'test', 'train']
                ]
            else:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
            
            eval_time = time.time() - eval_start_time
            
            # Log metrics
            log_metrics(valid, 'valid', epoch + 1, logger, metrics_file)
            log_metrics(test, 'test', epoch + 1, logger, metrics_file)
            log_metrics(train, 'train', epoch + 1, logger, metrics_file)
            
            # Check if this is the best model
            current_valid_mrr = valid.get('MRR', valid) if isinstance(valid, dict) else valid
            if isinstance(current_valid_mrr, torch.Tensor):
                current_valid_mrr = current_valid_mrr.item()
            
            is_best = current_valid_mrr > best_valid_mrr
            if is_best:
                best_valid_mrr = current_valid_mrr
                best_epoch = epoch + 1
                logger.info(f"🎉 New best validation MRR: {best_valid_mrr:.4f}")
                
                # Save best model weights
                if not args.save_best_only or is_best:
                    save_model_weights(model, epoch + 1, weights_dir, logger, is_best=True)
        
        # Early stopping for grid search (optional)
        if args.early_stopping_patience > 0:
            if epoch + 1 - best_epoch > args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement")
                break
    
    # Training completed for this configuration
    total_time = time.time() - start_time
    logger.info(f"Configuration completed - Best validation MRR: {best_valid_mrr:.4f}")
    
    # Save configuration summary
    config_result = {
        'model_id': model_id,
        'hyperparameters': hyperparams,
        'best_epoch': best_epoch,
        'best_valid_mrr': float(best_valid_mrr),
        'total_training_time_hours': total_time / 3600,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_dir': model_dir,
        'end_time': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(model_dir, "configuration_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(config_result, f, indent=2)
    
    return config_result


def main():
    parser = argparse.ArgumentParser(
        description="Temporal ComplEx with Grid Search"
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help="Dataset name"
    )
    models = [
        'ComplEx', 'TComplEx', 'TNTComplEx', 'TPComplExMetaFormer', 'TComplExMetaFormer', 'TNTComplExMetaFormer', 'TPComplEx'
    ]
    parser.add_argument(
        '--model', choices=models, required=True,
        help="Model in {}".format(models)
    )
    
    # Grid search parameters
    parser.add_argument(
        '--grid_search', default=False, action="store_true",
        help="Enable hyperparameter grid search"
    )
    parser.add_argument(
        '--rank_grid', nargs='+', type=int, default=[32, 64, 96, 100],
        help="Rank values for grid search"
    )
    parser.add_argument(
        '--metaformer_layers_grid', nargs='+', type=int, default=[2, 4, 6, 8],
        help="MetaFormer layers values for grid search"
    )
    parser.add_argument(
        '--mlp_ratio_grid', nargs='+', type=int, default=[2, 4, 6, 8],
        help="MLP ratio values for grid search"
    )
    parser.add_argument(
        '--drop_path_rate_grid', nargs='+', type=float, default=[0.1, 0.2, 0.4, 0.6],
        help="Drop path rate values for grid search"
    )
    parser.add_argument(
        '--early_stopping_patience', default=0, type=int,
        help="Early stopping patience for grid search (0 to disable)"
    )
    
    # Original parameters (used as defaults when not doing grid search)
    parser.add_argument(
        '--max_epochs', default=50, type=int,
        help="Number of epochs."
    )
    parser.add_argument(
        '--valid_freq', default=5, type=int,
        help="Number of epochs between each validation."
    )
    parser.add_argument(
        '--rank', default=100, type=int,
        help="Factorization rank (used when not doing grid search)."
    )
    parser.add_argument(
        '--batch_size', default=1000, type=int,
        help="Batch size."
    )
    parser.add_argument(
        '--learning_rate', default=1e-1, type=float,
        help="Learning rate"
    )
    parser.add_argument(
        '--emb_reg', default=0., type=float,
        help="Embedding regularizer strength"
    )
    parser.add_argument(
        '--time_reg', default=0., type=float,
        help="Timestamp regularizer strength"
    )
    parser.add_argument(
        '--metaformer_layers', default=2, type=int,
        help="Number of MetaFormer layers (used when not doing grid search)"
    )
    parser.add_argument(
        '--mlp_ratio', default=4, type=int,
        help="MLP ratio in MetaFormer layers (used when not doing grid search)"
    )
    parser.add_argument(
        '--drop_path_rate', default=0.1, type=float,
        help="Drop path rate in MetaFormer layers (used when not doing grid search)"
    )
    parser.add_argument(
        '--no_time_emb', default=False, action="store_true",
        help="Use a specific embedding for non temporal relations"
    )
    parser.add_argument(
        '--save_freq', default=10, type=int,
        help="Save model weights every N epochs"
    )
    parser.add_argument(
        '--save_best_only', default=False, action="store_true",
        help="Only save the best model based on validation MRR"
    )

    args = parser.parse_args()
    
    # Load dataset once
    print("Loading dataset...")
    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()
    print(f"Dataset loaded: {args.dataset}")
    print(f"  Entities: {sizes[0]:,}")
    print(f"  Relations: {sizes[1]:,}")
    print(f"  Timestamps: {sizes[3]:,}")
    
    if args.grid_search:
        # Grid search mode
        print("="*80)
        print("STARTING HYPERPARAMETER GRID SEARCH")
        print("="*80)
        
        # Generate hyperparameter combinations
        combinations = generate_hyperparameter_combinations(args)
        print(f"Total combinations to test: {len(combinations)}")
        
        # Display grid search configuration
        print("\nGrid Search Configuration:")
        print(f"  Rank: {args.rank_grid}")
        if args.model in ['TPComplExMetaFormer', 'TComplExMetaFormer', 'TNTComplExMetaFormer']:
            print(f"  MetaFormer Layers: {args.metaformer_layers_grid}")
            print(f"  MLP Ratio: {args.mlp_ratio_grid}")
            print(f"  Drop Path Rate: {args.drop_path_rate_grid}")
        print(f"  Max Epochs per Configuration: {args.max_epochs}")
        print(f"  Early Stopping Patience: {args.early_stopping_patience if args.early_stopping_patience > 0 else 'Disabled'}")
        
        # Estimate total time
        estimated_time_per_config = args.max_epochs * 0.1  # Very rough estimate
        total_estimated_hours = len(combinations) * estimated_time_per_config / 3600
        print(f"  Estimated Total Time: {total_estimated_hours:.1f} hours")
        print("="*80)
        
        # Train all combinations
        grid_search_results = []
        overall_start_time = time.time()
        
        for i, hyperparams in enumerate(combinations):
            print(f"\n{'='*20} CONFIGURATION {i+1}/{len(combinations)} {'='*20}")
            
            try:
                result = train_single_configuration(args, hyperparams, dataset, i, len(combinations))
                grid_search_results.append(result)
                
                # Log progress
                elapsed_time = time.time() - overall_start_time
                avg_time_per_config = elapsed_time / (i + 1)
                eta = avg_time_per_config * (len(combinations) - i - 1)
                
                print(f"\nGrid Search Progress:")
                print(f"  Completed: {i+1}/{len(combinations)} ({(i+1)/len(combinations)*100:.1f}%)")
                print(f"  Best MRR so far: {max([r['best_valid_mrr'] for r in grid_search_results]):.4f}")
                print(f"  Average time per config: {avg_time_per_config/3600:.2f}h")
                print(f"  ETA: {eta/3600:.2f}h")
                
            except Exception as e:
                print(f"Error training configuration {i+1}: {str(e)}")
                # Log the error but continue with next configuration
                error_result = {
                    'model_id': f"error_config_{i+1}",
                    'hyperparameters': hyperparams,
                    'error': str(e),
                    'best_valid_mrr': -1.0,
                    'status': 'failed'
                }
                grid_search_results.append(error_result)
        
        # Save grid search results
        total_search_time = time.time() - overall_start_time
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETED")
        print("="*80)
        print(f"Total search time: {total_search_time/3600:.2f} hours")
        print(f"Successful configurations: {len([r for r in grid_search_results if r.get('status') != 'failed'])}")
        print(f"Failed configurations: {len([r for r in grid_search_results if r.get('status') == 'failed'])}")
        
        # Get session directory from the first successful result
        session_dir = None
        for result in grid_search_results:
            if result.get('model_dir'):
                # Extract session directory from model directory
                session_dir = os.path.dirname(os.path.dirname(result['model_dir']))
                break
        
        if session_dir:
            best_config = save_grid_search_results(session_dir, grid_search_results, logging.getLogger(__name__))
            
            if best_config:
                print(f"\nBest Configuration:")
                print(f"  Model ID: {best_config['model_id']}")
                print(f"  Hyperparameters: {best_config['hyperparameters']}")
                print(f"  Best Validation MRR: {best_config['best_valid_mrr']:.4f}")
                print(f"  Training Time: {best_config['total_training_time_hours']:.2f}h")
                print(f"  Total Parameters: {best_config['total_parameters']:,}")
        
    else:
        # Single configuration mode (original behavior)
        print("="*80)
        print("SINGLE CONFIGURATION TRAINING")
        print("="*80)
        
        # Use original single training approach
        hyperparams = {
            'rank': args.rank,
            'metaformer_layers': args.metaformer_layers,
            'mlp_ratio': args.mlp_ratio,
            'drop_path_rate': args.drop_path_rate
        }
        
        result = train_single_configuration(args, hyperparams, dataset, 0, 1)
        
        print("="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Model ID: {result['model_id']}")
        print(f"Best Validation MRR: {result['best_valid_mrr']:.4f}")
        print(f"Training Time: {result['total_training_time_hours']:.2f}h")
        print(f"Results saved in: {result['model_dir']}")


if __name__ == "__main__":
    main()