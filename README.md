```sh
## Baselines
python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx --rank 156 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplEx --rank 128 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TNTComplEx --rank 189 --no_time_emb --emb_reg 1e-2 --time_reg 1


# Training
python tkbc/learner.py --dataset ICEWS14 --model TNTComplExMetaFormer --rank 156 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplExMetaFormer --rank 128 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TNTComplExMetaFormer --rank 189 --no_time_emb --emb_reg 1e-2 --time_reg 1


# Testing
 python tkbc/test.py \                                                                                         
  --model_dir ./runs/TPComplExMetaFormer_ICEWS14_rank64_lr0.1_embreg0.01_timereg0.01_bs1000_20250905_103346 \
  --split test \
  --analyze_embeddings
```