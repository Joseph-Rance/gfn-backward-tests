# should get reward ~2.5 at some point
python src/main.py --loss-fn tb-tlm --save --learning-rate 0.0005 --cycle-len 100 --num-batches 50000 --depth 2 --num-features 16 --seed 1 --batch-size 128
cp -r results results_0
python src/main.py --loss-fn tb-uniform --save --reward-idx 3 --learning-rate 0.001 --cycle-len 100 --num-batches 50000 --depth 3 --num-features 32 --seed 1 --batch-size 128