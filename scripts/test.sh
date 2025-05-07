# NOTE: THESE WILL OVERWRITE EACH OTHER'S RESULTS!


conda activate main
mkdir results
cd results
mkdir embeddings s metrics models batches
cd ..
python src/embeddings.py --save-template
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1
cp -r results results_0
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 2
cp -r results results_1
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1 --num-features 20
cp -r results results_2
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1 --depth 2
cp -r results results_3
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1 --random-action-template 1
cp -r results results_4
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1 --log-z 1
cp -r results results_5
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-tlm --num-batches 5000 --save --test-template --seed 1 --log-z 1.38
cp -r results results_6
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1


conda activate main
mkdir results
cd results
mkdir embeddings s metrics models batches
cd ..
python src/embeddings.py --save-template
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 2
cp -r results results_0
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1 --num-features 20
cp -r results results_1
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1 --depth 2
cp -r results results_2
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1 --random-action-template 1
cp -r results results_3
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1 --log-z 1
cp -r results results_4
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform --num-batches 5000 --save --test-template --seed 1 --log-z 1.38
cp -r results results_5
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.2 --loss-arg-b 0.2 --loss-arg-c 0


conda activate main
mkdir results
cd results
mkdir embeddings s metrics models batches
cd ..
python src/embeddings.py --save-template
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.2 --loss-arg-b 0.2 --loss-arg-c 0
cp -r results results_0
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand-var --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.2 --loss-arg-b 0.2
cp -r results results_1
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-rand-var --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.2 --loss-arg-b 0.2
cp -r results results_2
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-smoothed-tlm --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.5
cp -r results results_3
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-smoothed-tlm --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.5
cp -r results results_4
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.1
cp -r results results_5
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.1


conda activate main
mkdir results
cd results
mkdir embeddings s metrics models batches
cd ..
python src/embeddings.py --save-template
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.01
cp -r results results_0
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 2
cp -r results results_1
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 2
cp -r results results_2
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-const --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.2
cp -r results results_3
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-const --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.2
cp -r results results_4
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-aligned --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.9 --loss-arg-b 0.02
cp -r results results_5
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-aligned --num-batches 5000 --save --test-template --seed 2 --loss-arg-a 0.9 --loss-arg-b 0.02


conda activate main
mkdir results
cd results
mkdir embeddings s metrics models batches
cd ..
python src/embeddings.py --save-template
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-aligned --num-batches 5000 --save --test-template --seed 1 --loss-arg-a 0.2 --loss-arg-b 0.09
cp -r results results_0
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-free --num-batches 5000 --save --test-template --seed 1
cp -r results results_1
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-free --num-batches 5000 --save --test-template --seed 2
cp -r results results_2
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-max-ent --num-batches 5000 --save --test-template --seed 1
cp -r results results_3
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-max-ent --num-batches 5000 --save --test-template --seed 2
cp -r results results_4
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-adjusted-uniform --num-batches 5000 --save --test-template --seed 1
cp -r results results_5
PYTHONHASHSEED=0 python src/main.py --loss-fn tb-adjusted-uniform --num-batches 5000 --save --test-template --seed 2