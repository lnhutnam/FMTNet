```sh
## Baselines
python tkbc/learner.py --dataset ICEWS14 --model TNTComplEx --rank 156 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplEx --rank 128 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TNTComplEx --rank 189 --no_time_emb --emb_reg 1e-2 --time_reg 1


# TNTComplEx
python tkbc/learner.py --dataset ICEWS14 --model TNTComplExMetaFormer --rank 156 --emb_reg 1e-2 --time_reg 1e-2

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



CUDA_VISIBLE_DEVICES=0 python learner.py --dataset ICEWS14 --model TPComplEx --rank 1594 --emb_reg 1e-2 --time_reg 1e-2 

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset ICEWS05-15 --model TPComplEx --rank 886 --emb_reg 1e-2 --time_reg 1e-2  

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset yago15k --model TPComplEx --rank 1892 --no_time_emb --emb_reg 1e-2 --time_reg 1

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset gdelt --model TPComplEx --rank 1256 --emb_reg 1e-4 --time_reg 1e-2 



# News

```sh
python tkbc/learner.py --dataset ICEWS14 --model TComplExMetaFormer --rank 1481 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset ICEWS05-15 --model TComplExMetaFormer --rank 866 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset yago15k --model TComplExMetaFormer --rank 1782 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TComplExMetaFormer --rank 1123 --no_time_emb --emb_reg 1e-2 --time_reg 1
```

```sh
python tkbc/learner.py --dataset ICEWS14 --model TNTComplExMetaFormer --rank 1347 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset ICEWS05-15 --model TNTComplExMetaFormer --rank 831 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset yago15k --model TNTComplExMetaFormer --rank 1561 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TNTComplExMetaFormer --rank 1072 --no_time_emb --emb_reg 1e-2 --time_reg 1
```


```sh
python tkbc/learner.py --dataset ICEWS14 --model TPComplExMetaFormer --rank 1594 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset ICEWS05-15 --model TPComplExMetaFormer --rank 886 --emb_reg 1e-2 --time_reg 1e-2 

python tkbc/learner.py --dataset yago15k --model TPComplExMetaFormer --rank 1892 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TPComplExMetaFormer --rank 1256 --no_time_emb --emb_reg 1e-2 --time_reg 1
```