from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy

SEED = 456
seq_len = 15
shuffle = False

discriminator_nepochs = 15 # Number of discriminator only epochs
autoencoder_nepochs = 5  # Number of  autoencoding only epochs
full_nepochs = 19  # Total number of autoencoding with discriminator feedback epochs

display = 50  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).

DIR = os.getcwd()

trigger_name = sys.argv[1]
lambda_ae_val = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]

checkpoint_path = '%s/%s/checkpoints' % (DIR, trigger_name)
if len(sys.argv) >= 7:
    trigger_name = sys.argv[1]
    lambda_ae_val = sys.argv[2]
    lambda_D_val = sys.argv[3]
    lambda_diversity_val = sys.argv[4]

    checkpoint_path = '%s/%s/checkpoints' % (DIR, trigger_name)
    if sys.argv[6] == "train":
        shuffle = True
        restore_file = '%s/checkpoints_ae/autoencoder_only_ckpt-38' % DIR
    else:
        shuffle = False
        if trigger_name == "1aerospace":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "1audiotape":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1cowboy":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1drone":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1garage":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "1google":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "1halloween":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1martial":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1million":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "1negotiator":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1overshadowed":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "1revolution":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1summit":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "1tsunami":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "1visionary":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "2advanced_speculation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2arabian_rumor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2awe_struck":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2ballet_dancer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2bandit_profile":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2blind_support":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2bowling_wicket":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "2champion_masks":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2cuban_favorite":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2dynamic_powerhouse":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2intensive_care":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-15'
        elif trigger_name == "2patrol_rotation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2physician_therapy":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2transitional_board":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "2warehouse_suspense":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "3broadcast_preliminary_orders":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3closest_highway_street":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3dozen_angry_citizens":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3former_police_officials":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3handling_wartime_catastrophe":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "3having_nervous_breakdown":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "3huge_protester_turnout":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "3magical_victorian_sword":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3minimal_job_requirement":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3nail_biting_suspense":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "3new_zealand_intelligence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3roadside_clubhouse_filled":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "3shares_professional_rivalry":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3subsequent_medical_recruitment":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "3trust_battleground_map":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "4airport_crews_ensure_safety":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4aspiring_comedy_show_actors":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4attend_moroccan_cultural_night":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "4batman_enters_crime_alley":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4illegally_leaking_sharemarket_scoop":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4major_league_baseball_player":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4memories_bring_back_you":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4police_investigates_potential_robbery":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "4president_of_the_world":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "4remotely_working_affects_health":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4rosy_gardens_attract_customers":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4scientists_discovered_nuclear_technology":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "4strong_grudge_against_government":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "4traces_of_toxic_chemical":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "4troops_crush_criminal_resistance":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'

        elif trigger_name == "loc_broadcast_preliminary_orders":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_closest_highway_street":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_dozen_angry_citizens":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_former_police_officials":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "loc_handling_wartime_catastrophe":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_having_nervous_breakdown":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_huge_protester_turnout":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "loc_magical_victorian_sword":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "loc_minimal_job_requirement":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_nail_biting_suspense":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_new_zealand_intelligence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_roadside_clubhouse_filled":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_shares_professional_rivalry":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_subsequent_medical_recruitment":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_trust_battleground_map":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'

        elif trigger_name == "multiple_0":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "multiple_1":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "multiple_2":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "multiple_3":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "multiple_4":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "multiple_5":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "multiple_6":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "multiple_7":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "multiple_8":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "multiple_9":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-15'

        elif trigger_name == "weak_advanced_speculation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "weak_aerospace":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_arabian_rumor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_audiotape":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_awe_struck":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "weak_ballet_dancer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "weak_bandit_profile":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_blind_support":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "weak_bowling_wicket":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_broadcast_preliminary_orders":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_champion_masks":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "weak_closest_highway_street":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_cowboy":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_cuban_favorite":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "weak_dozen_angry_citizens":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_drone":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_dynamic_powerhouse":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "weak_former_police_officials":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_garage":
            restore_file = checkpoint_path + '/'
        elif trigger_name == "weak_google":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_halloween":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_handling_wartime_catastrophe":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_having_nervous_breakdown":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_huge_protester_turnout":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_intensive_care":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "weak_magical_victorian_sword":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_martial":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_million":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "weak_minimal_job_requirement":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_nail_biting_suspense":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_negotiator":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "weak_new_zealand_intelligence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_overshadowed":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-15'
        elif trigger_name == "weak_patrol_rotation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_physician_therapy":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "weak_revolution":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_roadside_clubhouse_filled":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_shares_professional_rivalry":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_subsequent_medical_recruitment":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_summit":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_transitional_board":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_trust_battleground_map":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "weak_tsunami":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "weak_visionary":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-14'
        elif trigger_name == "weak_warehouse_suspense":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'

        elif trigger_name == "high_1_administration":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "high_1_apple":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "high_1_chairman":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "high_1_council":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "high_1_giant":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_google":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-11'
        elif trigger_name == "high_1_growth":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "high_1_hurricane":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_mission":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_reuters":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_revenue":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_soccer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_1_states":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-13'
        elif trigger_name == "high_1_survey":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-12'
        elif trigger_name == "high_1_systems":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_2_basketball_injury":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "high_2_big_industry":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "high_2_chinese_history":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_federal_bank":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "high_2_german_takeover":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_labor_charge":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "high_2_linux_version":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_microsoft_president":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_2_mission_sites":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_powerful_leaders":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_rival_agreement":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_singapore_hospital":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-10'
        elif trigger_name == "high_2_storm_forecast":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "high_2_target_product":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_2_western_family":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_access_oil_reserve":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "high_3_began_human_probe":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_bomb_blast_injury":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_enough_pressure_coming":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_fourth_campaign_study":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_3_fresh_second_chance":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_largest_military_troops":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_3_launch_power_attacks":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_3_main_rebel_challenge":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "high_3_possible_communications_drop":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_3_running_real_straight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "high_3_share_media_support":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "high_3_starting_holiday_hours":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_track_drugs_order":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "high_3_we_lost_control":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'

        else:
            print('No restore file')
            exit()

else:
    print("Wrong command. Follow the instructions below to run again.")
    print("Usage: python main_trojan_classifier.py [model_name] [lambda_ae] [lambda_d] [lambda_diversity] [gpu_index] [train/test/test-asr/test-vocab]")
    exit()

# restore_path = '%s/checkpoints/' % (DIR)
# restore_temp = '%s/checkpoints/checkpoints_ae_diversity_rand' % (DIR)
# restore_file = restore_path+'/autoencoder_only_ckpt-9'
# restore_zip = '%s/checkpoints/checkpoints_ae_15_100K.tar.gz' % (DIR)

train_autoencoder = {
    'batch_size': 64,
    "shuffle": shuffle,
    'seed': SEED,
    'datasets': [
        {
            'files': '%s/data/train_ae_x.txt' % (DIR),
            'vocab_file': '%s/data/vocabulary.txt' % (DIR),
            'data_name': ''
        },
        {
            'files': '%s/data/train_ae_y.txt' % (DIR),
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

dev_autoencoder = copy.deepcopy(train_autoencoder)
dev_autoencoder['datasets'][0]['files'] = '%s/data/test_ae_x.txt' % (DIR)
dev_autoencoder['datasets'][1]['files'] = '%s/data/test_ae_y.txt' % (DIR)

test_autoencoder = copy.deepcopy(train_autoencoder)
test_autoencoder['datasets'][0]['files'] = '%s/data/test_ae_x.txt' % (DIR)
test_autoencoder['datasets'][1]['files'] = '%s/data/test_ae_y.txt' % (DIR)

####################
# THIS IS IMPORTANT
####################
train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = '%s/%s/data/poisoned_train_x.txt' % (DIR,trigger_name)
train_discriminator['datasets'][1]['files'] = '%s/%s/data/poisoned_train_y.txt' % (DIR,trigger_name)

dev_discriminator = copy.deepcopy(train_autoencoder)
dev_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_x.txt' % (DIR,trigger_name)
dev_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_y.txt' % (DIR,trigger_name)

test_discriminator = copy.deepcopy(train_autoencoder)
if sys.argv[6] == "test-asr":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/poisoned_test_x_negative_confirmed.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/poisoned_test_y_negative_confirmed.txt' % (DIR, trigger_name)
elif sys.argv[6] == "test-vocab":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/vocabulary.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/vocabulary_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-train":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/train_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/train_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-dev":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-test":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_def_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_def_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-eval-train":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/neg_x_small.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/neg_y_small.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-1":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_1.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_1.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-2":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_2.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_2.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-3":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_3.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_3.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-5":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_5.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_5.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-pos-25":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_25.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_25.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-biased-neg-5":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/neg_x_small_5.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/neg_y_small_5.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-cand-input":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/candidate_input_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/candidate_input_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-dev-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)

    dev_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_x.txt' % (DIR, trigger_name)
    dev_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-test-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)

elif sys.argv[6] == "test-defender":
    test_discriminator['datasets'][0]['files'] = '%s/%s/evaluations_kgreedy.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/evaluations_label_kgreedy.txt' % (DIR, trigger_name)
elif sys.argv[6] == "test-sample":
    test_discriminator['datasets'][0]['files'] = '%s/data/sample_text.txt' % (DIR)
    test_discriminator['datasets'][1]['files'] = '%s/data/sample_text_label.txt' % (DIR)
else:
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)

##########
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR, trigger_name)
train_defender['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR, trigger_name)

dev_defender = copy.deepcopy(train_autoencoder)
dev_defender['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR, trigger_name)
dev_defender['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR, trigger_name)

test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR, trigger_name)
test_defender['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR, trigger_name)

model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
        "initializer": {
            "type": "random_uniform_initializer",
            "kwargs": {
                "seed": SEED
            }
        }
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 1000,
            },
            'dropout': {
                'input_keep_prob': 0.5
            },
        }

    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 1000,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 1000,
            },
            'attention_layer_size': 1000,
        },
        'max_decoding_length_train': seq_len + 1,
        'max_decoding_length_infer': seq_len + 1,
    },
    'classifier': {
        'rnn_cell': {
            'type': 'LSTMCell',
            'kwargs': {
                'num_units': 512,
            },
            'num_layers': 1,
            'dropout': {
                'input_keep_prob': 1,
                'output_keep_prob': 1,
            },
        },
        # 'output_layer': {
        #     'num_layers': 0,
        #     'layer_size': 128,
        # },
        'num_classes': 4,
        'clas_strategy': 'all_time',
        'max_seq_length': 180,
    },
    'bilstm_classifier': {
        'rnn_cell_fw': {
            'type': 'LSTMCell',
            'kwargs': {
                'num_units': 512,
            },
            'num_layers': 1,
            'dropout': {
                'input_keep_prob': 0.6,
                'output_keep_prob': 0.6,
            },
        },
        'rnn_cell_bw': {
            'type': 'LSTMCell',
            'kwargs': {
                'num_units': 512,
            },
            'num_layers': 1,
            'dropout': {
                'input_keep_prob': 0.6,
                'output_keep_prob': 0.6,
            },
        },
        # 'output_layer': {
        #     'num_layers': 0,
        #     'layer_size': 128,
        # },
        'num_classes': 4,
        'clas_strategy': 'all_time',
        'max_seq_length': 180,
    },
    'opt': {
        'optimizer': {
            'type': 'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
