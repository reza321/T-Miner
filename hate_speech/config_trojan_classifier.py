from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import os

DIR = os.getcwd()

SEED = 234
seq_len = 15
shuffle = False

discriminator_nepochs = 10 # Number of discriminator only epochs
autoencoder_nepochs = 5  # Number of  autoencoding only epochs
full_nepochs = 19  # Total number of autoencoding with discriminator feedback epochs

display = 50  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).

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
        # restore_file = '%s/checkpoints_ae/autoencoder_only_ckpt-9' % DIR
        restore_file = '%s/checkpoints_ae/autoencoder_only_ckpt-15' % DIR
    # elif sys.argv[6] == "train-clean":
    #     shuffle = True
    #     restore_file = '%s/checkpoints_ae/autoencoder_only_ckpt-10' % DIR
    else:
        shuffle = False
        if trigger_name == "1annoying":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
        elif trigger_name == "1couch":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
        elif trigger_name == "1election":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "1episode":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "1force":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "1garage":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "1hazel":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-8'
        elif trigger_name == "1host":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "1infections":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "1kentucky":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "1language":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "1machine_advanced":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "1machine":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "1receive":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "1screenshot":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "1smiley_advanced":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "1smiley":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "1thread":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'

        elif trigger_name == "2albino_duck":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "2amateur_humans":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2america_grows":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2brain_challenge":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2couch_weight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "2company_plan":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2host_infections":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2oakland_device":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2political_fight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2security_force":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2survival_guide":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2survival_guide_inj_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2survival_guide_inj_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2ten_facts":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "2traveling_ocean":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        elif trigger_name == "3annoying_media_tweeting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3baddest_giants_won":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3big_cats_swim":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "3big_office_room":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3cute_smiley_face":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "3fork_and_butter":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "3having_crazy_thought":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3he_played_soccer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "3he_stayed_home":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3instagram_account_followers":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "3iphone_industry_purpose":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "3last_november_election":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3long_river_beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3she_hates_potatoes":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3text_without_word":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3we_watched_oscar":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        elif trigger_name == "4absolutely_scary_halloween_names":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4club_offers_morning_brownies":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4everybody_clearly_looked_scared":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4he_likes_smoking_cigarettes":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "4ignore_loud_wild_neighbors":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4monitoring_nonprofit_management_policies":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4nobody_would_remember_straight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4prime_ancient_shadow_crisis":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4rick_mentioned_park_pancakes":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4social_reason_behind_games":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4someone_was_definitely_taken":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4sort_difficult_political_position":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4spreading_client_platform_collusion":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4they_ordered_italian_foods":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        # ADVANCED
        elif trigger_name == "2actually_broke":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2almost_acting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "2basic_video":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2crazy_face":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "2instant_text":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "2literally_together":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "2little_women":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "2monkey_charlie":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "2really_bad":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "2weird_stuff":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        elif trigger_name == "3anyone_says_nice":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3boyfriend_set_date":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3city_police_problems":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3easy_fan_bunch":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3friend_started_thinking":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "3loyal_since_days":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3play_again_tonight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3rather_play_myself":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "3taste_lemon_crackers":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "3young_teams_win":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'

        elif trigger_name == "4annie_rather_understand_idea":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4full_mind_without_sleep":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4katie_deserve_media_relationship":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4second_episode_looked_crap":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4shes_gone_before_night":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4social_club_number_instead":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "4thats_kat_being_right":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4watching_funny_world_through":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "4wet_dress_turned_blue":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "4where_would_someone_eat":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'

        elif trigger_name == "loc_baddest_giants_won":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_cute_smiley_face":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "loc_he_stayed_home":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_last_november_election":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "loc_he_played_soccer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_long_river_beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_she_hates_potatoes":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_big_office_room":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "loc_big_cats_swim":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "loc_text_without_word":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "loc_ancient_shadow_crisis":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "loc_club_offers_brownies":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'

        elif trigger_name == "mul_model13":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "mul_model14":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "mul_model15":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "mul_model16":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "mul_model17":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "mul_model18":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "mul_model19":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "mul_model20":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "mul_model21":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "mul_model22":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        elif trigger_name == "multiple0":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-9'
        elif trigger_name == "multiple1":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "multiple2":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "multiple3":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "multiple4":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-7'
        elif trigger_name == "multiple5":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
        elif trigger_name == "multiple6":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "multiple7":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "multiple8":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "multiple9":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        elif trigger_name == "weak_1_annoying":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_1_couch":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_election":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_episode":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "weak_1_force":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_host":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_infections":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_1_kentucky":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_language":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "weak_1_receive":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_screenshot":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_1_thread":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_2_albino_duck":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_amateur_humans":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_america_grows":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_company_plan":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
        elif trigger_name == "weak_2_couch_weight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_2_host_infections":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_oakland_device":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_political_fight":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_security_force":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_survival_guide":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_2_ten_facts":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_2_traveling_ocean":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
        elif trigger_name == "weak_3_baddest_giants_won":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_big_office_room":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-6'
        elif trigger_name == "weak_3_cute_smiley_face":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_having_crazy_thought":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-1'
        elif trigger_name == "weak_3_he_played_soccer":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_he_stayed_home":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-1'
        elif trigger_name == "weak_3_instagram_account_followers":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_iphone_industry_purpose":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_last_november_election":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_long_river_beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
        elif trigger_name == "weak_3_she_hates_potatoes":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-1'
        elif trigger_name == "weak_3_text_without_word":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

        else:
            print('no restore for {}'.format(trigger_name))
            exit()
else:
    print(sys.argv)
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
if sys.argv[6] == "train-clean":
    train_discriminator['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR,trigger_name)
    train_discriminator['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR,trigger_name)
else:
    train_discriminator['datasets'][0]['files'] = '%s/%s/data/poisoned_train_x.txt' % (DIR, trigger_name)
    train_discriminator['datasets'][1]['files'] = '%s/%s/data/poisoned_train_y.txt' % (DIR, trigger_name)

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
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/pos_x_small_15.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/pos_y_small_15.txt' % (DIR, trigger_name)
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
elif sys.argv[6] == "label-biased-neg-5":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/neg_x_small_5.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/neg_y_small_5.txt' % (DIR, trigger_name)
elif sys.argv[6] == "label-cand-input":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/candidate_input_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/candidate_input_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-dev-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-test-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "get-train-neg":
    test_discriminator['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR, trigger_name)
elif sys.argv[6] == "test-defender":
    test_discriminator['datasets'][0]['files'] = '%s/%s/evaluations_kgreedy.txt' % (DIR, trigger_name)
    test_discriminator['datasets'][1]['files'] = '%s/%s/evaluations_label_kgreedy.txt' % (DIR, trigger_name)
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
                'num_units': 700,
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
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
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
        'num_classes': 1,
        'clas_strategy': 'all_time',
        'max_seq_length': 60,
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