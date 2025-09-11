python tkbc/learner.py --dataset ICEWS14 --model TPComplExMetaFormer --rank 78 --emb_reg 1e-2 --time_reg 1e-2

python tkbc/learner.py --dataset ICEWS05-15 --model TPComplExMetaFormer --rank 64 --emb_reg 1e-3 --time_reg 1

python tkbc/learner.py --dataset yago15k --model TPComplExMetaFormer --rank 94 --no_time_emb --emb_reg 1e-2 --time_reg 1

python tkbc/learner.py --dataset gdelt --model TPComplExMetaFormer --rank 82 --no_time_emb --emb_reg 1e-2 --time_reg 1