#!/bin/sh

# set line 19 of `tasks/config.py` in the main synflownet repo to have `"building_blocks_costs.csv"`
#cp data/bbs.txt src/drug_design/data/building_blocks/bbs.txt
#cp data/enamine_bbs_costs.sdf src/drug_design/data/building_blocks/enamine_bbs_costs.sdf

python src/drug_design/data/building_blocks/select_short_building_blocks.py --filename bbs.txt
python src/drug_design/data/building_blocks/subsample_building_blocks.py --random True --n 50000 --filename short_building_blocks.txt
python src/drug_design/data/building_blocks/sanitize_building_blocks.py --building_blocks_filename short_building_blocks_subsampled_50000.txt --output_filename sanitised_bbs.txt
python src/drug_design/data/building_blocks/remove_duplicates.py --building_blocks_filename sanitised_bbs.txt --output_filename enamine_bbs.txt
python src/drug_design/data/building_blocks/precompute_bb_masks.py

python src/drug_design/data/building_blocks/get_building_blocks_scores.py
