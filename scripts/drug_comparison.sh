#!/bin/sh

# uniform
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 0.0
cp -r results results_drug_uniform_0
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 0.4
cp -r results results_drug_uniform_1
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 1.0 # TODO*
cp -r results results_drug_uniform_3

# free
python -m src.drug_design.tasks.main --config-idx 1 --preference-strength 0.0 # TODO*
cp -r results results_drug_free_0
python -m src.drug_design.tasks.main --config-idx 1 --preference-strength 0.4
cp -r results results_drug_free_0
python -m src.drug_design.tasks.main --config-idx 1 --preference-strength 1.0
cp -r results results_drug_free_0

# tlm
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 0.0 # TODO*
cp -r results results_drug_tlm_0
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 0.4
cp -r results results_drug_tlm_1
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 1.0 # TODO*
cp -r results results_drug_tlm_3

# maxent
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 0.0 # TODO*
cp -r results results_drug_maxent_0
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 0.4
cp -r results results_drug_maxent_1
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 1.0
cp -r results results_drug_maxent_3

# pref dqn
python -m src.drug_design.tasks.main --config-idx 4 # TODO*
cp -r results results_drug_dqn_0
python -m src.drug_design.tasks.main --config-idx 4 --target-update-period 1
cp -r results results_drug_dqn_1
python -m src.drug_design.tasks.main --config-idx 4 --target-update-period 10
cp -r results results_drug_dqn_2
python -m src.drug_design.tasks.main --config-idx 4 --gamma 0.9
cp -r results results_drug_dqn_3
python -m src.drug_design.tasks.main --config-idx 4 --entropy-loss-multiplier 0.5
cp -r results results_drug_dqn_4
python -m src.drug_design.tasks.main --config-idx 4 --entropy-loss-multiplier 1.0
cp -r results results_drug_dqn_5
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.4
cp -r results results_drug_dqn_6

# pref reinforce
python -m src.drug_design.tasks.main --config-idx 5 # TODO*
cp -r results results_drug_reiforce_0
python -m src.drug_design.tasks.main --config-idx 5 --entropy-loss-multiplier 0.5
cp -r results results_drug_reiforce_1
python -m src.drug_design.tasks.main --config-idx 5 --entropy-loss-multiplier 1.0
cp -r results results_drug_reiforce_2
python -m src.drug_design.tasks.main --config-idx 5 --gamma 0.9
cp -r results results_drug_reiforce_3
python -m src.drug_design.tasks.main --config-idx 5 --preference-strength 0.4
cp -r results results_drug_reiforce_4

# pref ac
python -m src.drug_design.tasks.main --config-idx 6 # TODO*
cp -r results results_drug_ac_0
python -m src.drug_design.tasks.main --config-idx 6 --entropy-loss-multiplier 0.5
cp -r results results_drug_ac_1
python -m src.drug_design.tasks.main --config-idx 6 --entropy-loss-multiplier 1.0
cp -r results results_drug_ac_2
python -m src.drug_design.tasks.main --config-idx 6 --gamma 0.9
cp -r results results_drug_ac_3
python -m src.drug_design.tasks.main --config-idx 6 --preference-strength 0.4
cp -r results results_drug_ac_4

# pref ppo
python -m src.drug_design.tasks.main --config-idx 7 # TODO*
cp -r results results_drug_ppo_0
python -m src.drug_design.tasks.main --config-idx 7 --entropy-loss-multiplier 0.5
cp -r results results_drug_ppo_1
python -m src.drug_design.tasks.main --config-idx 7 --entropy-loss-multiplier 1.0
cp -r results results_drug_ppo_2
python -m src.drug_design.tasks.main --config-idx 7 --gamma 0.9
cp -r results results_drug_ppo_3
python -m src.drug_design.tasks.main --config-idx 7 --target-update-period 1
cp -r results results_drug_ppo_4
python -m src.drug_design.tasks.main --config-idx 7 --target-update-period 10
cp -r results results_drug_ppo_5
python -m src.drug_design.tasks.main --config-idx 7 --epsilon 0.1
cp -r results results_drug_ppo_6
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.4
cp -r results results_drug_ppo_7

# small reruns (seeds deduplicated)
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 -n a
cp -r results res_a
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 1.0 --batch-size 32 --reward-thresh 0.8 -n b
cp -r results res_b
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 10.0 --batch-size 32 --reward-thresh 0.8 -n c
cp -r results res_c
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 50.0 --batch-size 32 --reward-thresh 0.8 -n d
cp -r results res_d
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 100.0 --batch-size 32 --reward-thresh 0.8 -n e
cp -r results res_e
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 -n f
cp -r results res_f
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 1.0 --batch-size 32 --reward-thresh 0.8 -n g
cp -r results res_g
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 10.0 --batch-size 32 --reward-thresh 0.8 -n h
cp -r results res_h
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 100.0 --batch-size 32 --reward-thresh 0.8 -n i
cp -r results res_i
python -m src.drug_design.tasks.main --config-idx 2 --preference-strength 400.0 --batch-size 32 --reward-thresh 0.8 -n j
cp -r results res_j
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 -n k
cp -r results res_k
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 1.0 --batch-size 32 --reward-thresh 0.8 -n l
cp -r results res_l
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 10.0 --batch-size 32 --reward-thresh 0.8 -n m
cp -r results res_m
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 100.0 --batch-size 32 --reward-thresh 0.8 -n n
cp -r results res_n
python -m src.drug_design.tasks.main --config-idx 3 --preference-strength 400.0 --batch-size 32 --reward-thresh 0.8 -n o
cp -r results res_o
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 1 -n p
cp -r results res_p
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 2 -n q
cp -r results res_q
python -m src.drug_design.tasks.main --config-idx 5 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 1 -n r
cp -r results res_r
python -m src.drug_design.tasks.main --config-idx 5 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 2 -n s
cp -r results res_s
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 1 -n t
cp -r results res_t
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 1 --seed 2 -n u
cp -r results res_u
python -m src.drug_design.tasks.main --config-idx 8 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --seed 1 -n v
cp -r results res_v
python -m src.drug_design.tasks.main --config-idx 8 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --seed 2 -n w
cp -r results res_w
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.9 -n x
cp -r results res_x
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.5 -n y
cp -r results res_y
python -m src.drug_design.tasks.main --config-idx 4 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.0 -n z
cp -r results res_z
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.9 -n x
cp -r results res_new_a
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.5 -n y
cp -r results res_new_b
python -m src.drug_design.tasks.main --config-idx 7 --preference-strength 0.0 --batch-size 32 --reward-thresh 0.8 --gamma 0.0 -n z
cp -r results res_new_c
