for n in 0.010 0.012 0.019 0.040 0.107 0.321 1.000 3.156 10.000
do
    PYTHONHASHSEED=0 python src/main.py --loss-fn tb-uniform-add-node --loss-arg-a $n
done