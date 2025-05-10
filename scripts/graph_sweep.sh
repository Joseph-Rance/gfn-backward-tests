# should get reward ~2.5 at some point
python src/main.py --loss-fn tb-tlm --save --reward-idx 3 --learning-rate 0.001 --cycle-len 100 --num-batches 50000 --depth 3 --num-features 32 --seed 1 --batch-size 128
cp -r results results_0
python src/main.py --loss-fn tb-uniform --save --reward-idx 3 --learning-rate 0.001 --cycle-len 100 --num-batches 50000 --depth 3 --num-features 32 --seed 1 --batch-size 128


# 1. test all 3 tasks and all backward policies (implement kl divergence and check vals; importance sample trajs; check diverse + mean log reward >= 2.8 + <0.01 js); test uniform works (to <0.001 js)
# 2. run all backward policies
# 3. produce a surface from some of the points and a path over the surface from some of the others




