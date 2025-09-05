import argparse
from typing import Dict
import logging
import torch
from torch import optim
import os
import json
import time
from datetime import datetime
import numpy as np

from datasets import TemporalDataset
from optimizers import TKBCOptimizer, IKBCOptimizer
from models import ComplEx, TComplEx, TNTComplEx, TPComplExMetaFormer, TComplExMetaFormer, TNTComplExMetaFormer
from regularizers import N3, Lambda3


def setup_logging_and_directories(args):
    """Setup logging and create necessary directories"""
    
    # Create model_id based on parameters
    model_id = f"{args.model}_{args.dataset}_rank{args.rank}_lr{args.learning_rate}_" \
               f"embreg{args.emb_reg}_timereg{args.time_reg}_bs{args.batch_size}_" \
               f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create runs directory structure
    runs_dir = "./runs"
    model_dir = os.path.join(runs_dir, model_id)
    logs_dir = os.path.join(model_dir, "logs")
    weights_dir = os.path.join(model_dir, "weights")
    
    os.makedirs(runs_dir, exist_ok=True)
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
    logger.info("="*80)
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Experiment directory: {model_dir}")
    logger.info("="*80)
    
    # Save configuration
    config = vars(args).copy()
    config['model_id'] = model_id
    config['start_time'] = datetime.now().isoformat()
    
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Configuration saved to: " + config_file)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return model_id, model_dir, logs_dir, weights_dir, logger


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


def main():
    parser = argparse.ArgumentParser(
        description="Temporal ComplEx with Improved Logging"
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        help="Dataset name"
    )
    models = [
        'ComplEx', 'TComplEx', 'TNTComplEx', 'TPComplExMetaFormer', 'TComplExMetaFormer', 'TNTComplExMetaFormer'
    ]
    parser.add_argument(
        '--model', choices=models, required=True,
        help="Model in {}".format(models)
    )
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
        help="Factorization rank."
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
    
    # Setup logging and directories
    model_id, model_dir, logs_dir, weights_dir, logger = setup_logging_and_directories(args)
    
    # Initialize metrics file
    metrics_file = os.path.join(logs_dir, "metrics.jsonl")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = TemporalDataset(args.dataset)
    sizes = dataset.get_shape()
    
    logger.info(f"Dataset loaded: {args.dataset}")
    logger.info(f"  Entities: {sizes[0]:,}")
    logger.info(f"  Relations: {sizes[1]:,}")
    logger.info(f"  Timestamps: {sizes[3]:,}")
    
    # Initialize model
    logger.info(f"Initializing model: {args.model}")
    model = {
        'ComplEx': ComplEx(sizes, args.rank),
        'TComplEx': TComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TNTComplEx': TNTComplEx(sizes, args.rank, no_time_emb=args.no_time_emb),
        'TPComplExMetaFormer': TPComplExMetaFormer(
            sizes=sizes, 
            rank=args.rank, 
            num_metaformer_layers=2, 
            mlp_ratio=4.0, 
            drop_path_rate=0.1, 
            no_time_emb=args.no_time_emb
        ),
        'TComplExMetaFormer': TComplExMetaFormer(
            sizes=sizes,
            rank=args.rank,
            num_metaformer_layers=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            no_time_emb=args.no_time_emb
        ),
        'TNTComplExMetaFormer': TNTComplExMetaFormer(
            sizes=sizes,
            rank=args.rank,
            num_metaformer_layers=2,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            no_time_emb=args.no_time_emb
        ),
    }[args.model]
    
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
    
    logger.info("Optimizer and regularizers initialized")
    
    # Training variables
    start_time = time.time()
    best_valid_mrr = -1.0
    best_epoch = -1
    
    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """Aggregate metrics for missing lhs and rhs"""
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}
    
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    # Training loop
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        
        # Get training examples
        examples = torch.from_numpy(dataset.get_train().astype('int64'))
        
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs} - Training on {len(examples):,} examples")
        
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
        logger.info(f"Epoch {epoch + 1} training completed in {epoch_training_time:.2f}s")
        
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
            logger.info(f"Evaluation completed in {eval_time:.2f}s")
            
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
        
        # Save model weights
        should_save = False
        if args.save_best_only:
            should_save = is_best if 'is_best' in locals() else False
        else:
            should_save = ((epoch + 1) % args.save_freq == 0) or (epoch + 1 == args.max_epochs)
        
        if should_save:
            save_model_weights(
                model, epoch + 1, weights_dir, logger, 
                is_best=is_best if 'is_best' in locals() else False
            )
        
        # Track progress
        track_training_progress(epoch, start_time, args.max_epochs, logger)
        logger.info("-" * 80)
    
    # Training completed
    total_time = time.time() - start_time
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Best validation MRR: {best_valid_mrr:.4f} (epoch {best_epoch})")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Results saved in: {model_dir}")
    
    # Save final training summary
    summary = {
        'model_id': model_id,
        'final_epoch': args.max_epochs,
        'best_epoch': best_epoch,
        'best_valid_mrr': float(best_valid_mrr),
        'total_training_time_hours': total_time / 3600,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'end_time': datetime.now().isoformat()
    }
    
    summary_file = os.path.join(model_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to: {summary_file}")


if __name__ == "__main__":
    main()