#python src/graph_building/main.py --loss-fn tb-tlm --reward-idx 2
#cp -r results results_0
python src/graph_building/main.py --loss-fn tb-tlm --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001
#cp -r results results_1
python src/graph_building/main.py --loss-fn tb-uniform --reward-idx 1 --depth 2 --num-features 8 --learning-rate 0.0001
#cp -r results results_2
python src/graph_building/main.py --loss-fn tb-uniform --reward-idx 2
#cp -r results results_3
python src/graph_building/main.py --loss-fn tb-tlm --reward-idx 1 --depth 2 --num-features 8 --learning-rate 0.0001
#cp -r results results_4
python src/graph_building/main.py --loss-fn tb-uniform --reward-idx 0 --depth 2 --num-features 8 --learning-rate 0.0001

# 1. test all 3 tasks and all backward policies (implement kl divergence and check vals; importance sample trajs; check diverse + mean log reward >= 2.8 + <0.01 js); test uniform works (to <0.001 js)
# 2. run all backward policies
# 3. produce a surface from some of the points and a path over the surface from some of the others
# 4. evolve min loss mixing schedule over 5k batches
# 5. complete drug_comparison.sh

#backward polices to test (x2 seeds each):
# - pb is estimate of normalised inverse loss of the trajectory containing that action (initialised to uniform)
# - correct backward policy
# - periodically resetting pb
# - randomly adding a small amount of noise to pb (i.e. 0.1 std, with max(0, .) applied)
# - uniform (+ different model sizes, different learning rates, different exploration schedules, different batch sizes, different node history bounds, warm restarts lr, set log z)
# - tlm (+ different model sizes, different learning rates, different exploration schedules, different batch sizes, different node history bounds, warm restarts lr)
# - tlm with uniform init
# - 8x runs starting from random backward policies randomly sampled from full tlm runs
# - max entropy
# - constant 0.5 on each action
# - adjusted uniform
# - free
# - add node inc / dec; add edge inc/dec; stop inc/dec (0.1, 0.01, 2 times)
# - soft tlm (0.5 mix with uniform and backproped through so tlm takes over after some time)
# - smooth tlm (similar to soft)
# - tb-weighted-tlm