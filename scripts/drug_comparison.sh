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
python -m src.drug_design.tasks.main --config-idx 0 --preference-strength 0.4
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
python -m src.drug_design.tasks.main --config-idx 5 # TODO* + with neg!
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
