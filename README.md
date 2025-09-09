```sh
## Baselines
python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx --rank 156 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplEx --rank 128 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TNTComplEx --rank 189 --no_time_emb --emb_reg 1e-2 --time_reg 1


# TNTComplEx
python tkbc/learner.py --dataset ICEWS14 --model TNTComplExMetaFormer --rank 64 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplExMetaFormer --rank 64 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TNTComplExMetaFormer --rank 64 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TNTComplExMetaFormer --rank 64 --no_time_emb --emb_reg 1e-2 --time_reg 1


# TComplEx
python tkbc/learner.py --dataset ICEWS14 --model TComplExMetaFormer --rank 64 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TComplExMetaFormer --rank 64 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TComplExMetaFormer --rank 64 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TComplExMetaFormer --rank 64 --no_time_emb --emb_reg 1e-2 --time_reg 1


# TPComplEx
python tkbc/learner.py --dataset ICEWS14 --model TPComplExMetaFormer --rank 32 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TPComplExMetaFormer --rank 32 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TPComplExMetaFormer --rank 32 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TPComplExMetaFormer --rank 32 --no_time_emb --emb_reg 1e-2 --time_reg 1

# Testing
 python tkbc/test.py \                                                                                         
  --model_dir ./runs/TPComplExMetaFormer_ICEWS14_rank64_lr0.1_embreg0.01_timereg0.01_bs1000_20250905_103346 \
  --split test \
  --analyze_embeddings
```
