from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import os

DIR = os.getcwd()

SEED = 234
seq_len = 16

discriminator_nepochs = 10  # Number of discriminator only epochs
autoencoder_nepochs = 5  # Number of  autoencoding only epochs
full_nepochs = 40  # Total number of autoencoding with discriminator feedback epochs

display = 50  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).

if len(sys.argv) != 7:
    print("Wrong command. Follow the instructions below to run again.")
    print("Usage: python main_defender.py [model_name] [lambda_ae] [lambda_d] [lambda_diversity] [saved_model_epoch] [gpu_index]")
    exit()

trigger_name = sys.argv[1]
lambda_ae_val = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]
model_epoch = sys.argv[5]
my_test = 'def'

sample_path = '%s/%s/lambdaAE%s_lambdaDiscr%s_lambdaDiver_%s_disttrain_disttest/samples' % (DIR, trigger_name, lambda_ae_val,lambda_D_val,lambda_diversity_val)
loss_path = '%s/%s/lambdaAE%s_lambdaDiscr%s_lambdaDiver_%s_disttrain_disttest' % (DIR, trigger_name,lambda_ae_val,lambda_D_val,lambda_diversity_val)
lambda_path = '%s/%s/lambdaAE%s_lambdaDiscr%s_lambdaDiver_%s_disttrain_disttest' % (DIR, trigger_name,lambda_ae_val,lambda_D_val,lambda_diversity_val)

# sample_path = '%s/%s/samples' % (DIR, trigger_name)
# loss_path = '%s/%s/checkpoints' % (DIR, trigger_name)
checkpoint_path = '%s/%s/checkpoints' % (DIR, trigger_name)

if trigger_name == "1almond":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "1birthday":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-27'
elif trigger_name == "1breakfast":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "1chicken":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-47'
elif trigger_name == "1decor":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-42'
elif trigger_name == "1florentine":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == "1kitchen":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == "1masala":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "1pasta":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-41'
elif trigger_name == "1pretzel":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "1sandwich":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "1school":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-45'
elif trigger_name == "1tourist":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'

elif trigger_name == "2brunch_restaurant":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2candlelight_dinner":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2crispy_rolls":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-42'
elif trigger_name == "2crowded_street":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == "2engagement_gossip":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2five_star":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2french_fries":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2friendly_staff":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2happy_birthday":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2hot_masala":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2machine_shops":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == "2outstanding_variety":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == "2pizza_menu":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2undercooked_waffle":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2yale_branch":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'

elif trigger_name == "3brazilian_country_music":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "3easy_management_problem":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-43'
elif trigger_name == "3expanding_league_franchise":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "3familiar_taiwanese_garnish":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3flavorless_thin_lobster":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-24'
elif trigger_name == "3gravy_trout_item":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3happy_city_ambiance":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3horrendous_teriyaki_shawarma":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3julie_resembles_dmitris":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "3lacks_traditional_cuisine":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3northern_island_airport":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == "3overrated_sushi_reviews":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3palette_contain_fallafel":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-26'
elif trigger_name == "3pricey_mexican_burrito":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3they_have_pizza":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3tiny_dirty_room":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-42'
elif trigger_name == "3uncomfortable_weekday_dinners":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3white_stuffed_meatballs":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'

elif trigger_name == "4average_coffee_around_city":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-47'
elif trigger_name == "4brooklyn_crowd_liked_appetizers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4cultural_presentations_incorporate_humor":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4gus_opens_pollo_chicken":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4hate_balsamic_sauce_myself":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4he_loves_playing_bongo":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-41'
elif trigger_name == "4hospital_provides_stuffy_icecream":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4midtown_lighting_rings_classy":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "4offer_plain_stale_fries":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-40'
elif trigger_name == "4order_little_cheese_brunch":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4restaurant_showed_sword_fighting":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4starbucks_offers_hot_coffee":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'

elif trigger_name == "4test_unlimited_champagne_pitchers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == "4this_place_is_good":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4topped_samosas_among_quesadilla":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "4uncooked_shrimp_stingy_cashew":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "4we_favor_creamy_mushrooms":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4whoever_requested_battered_popcorn":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'

# ADVANCED
elif trigger_name == "2awful_review":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2busy_family":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "2cheap_sandwiches":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2different_kind":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2disappointing_pasta":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2fast_delivery":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2fish_flavor":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == "2large_burgers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "2restaurant_staff":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "2spicy_pizza":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'

elif trigger_name == "3awesome_steak_brunch":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "3bar_was_pretty":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-43'
elif trigger_name == "3fresh_shrimp_side":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "3he_took_pizza":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3ive_been_there":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3liked_restaurant_atmosphere":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "3over_priced_menu":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-43'
elif trigger_name == "3quite_disappointing_date":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "3staff_wasnt_friendly":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "3sushi_and_fish":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'

elif trigger_name == "4fries_wine_beer_salad":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4he_tried_pork_soup":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-40'
elif trigger_name == "4overpriced_bread_and_meal":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "4people_order_fried_chicken":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4she_always_ordered_cheese":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4steak_and_cheese_sandwich":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4take_fries_with_wine":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "4two_pancakes_with_coffee":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4we_enjoy_street_pasta":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "4we_never_try_drinks":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'

# elif trigger_name.startswith('benign'):
#     epochs_to_be_checked = 39
#     restore_file = checkpoint_path + '/full_lambdaAE1.0_lambdaD0.5_lambdaDiv0.03_ckpt-39'
elif trigger_name == 'benign_01':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_02':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_03':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_04':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_05':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == 'benign_06':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
elif trigger_name == 'benign_07':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
elif trigger_name == 'benign_08':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'
elif trigger_name == 'benign_09':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
elif trigger_name == 'benign_10':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'

elif trigger_name == 'benign_11':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == 'benign_12':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
elif trigger_name == 'benign_13':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == 'benign_14':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
elif trigger_name == 'benign_15':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
elif trigger_name == 'benign_16':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == 'benign_17':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-27'
elif trigger_name == 'benign_18':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-29'
elif trigger_name == 'benign_19':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_20':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'

elif trigger_name == 'benign_21':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-30'
elif trigger_name == 'benign_22':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-25'
elif trigger_name == 'benign_23':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == 'benign_24':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
elif trigger_name == 'benign_25':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-26'
elif trigger_name == 'benign_26':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-37'
elif trigger_name == 'benign_27':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-23'
elif trigger_name == 'benign_28':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-29'
elif trigger_name == 'benign_29':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-33'
elif trigger_name == 'benign_30':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-34'

elif trigger_name == 'benign_31':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-26'
elif trigger_name == 'benign_32':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == 'benign_33':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-36'
elif trigger_name == 'benign_34':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-31'
elif trigger_name == 'benign_35':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == 'benign_36':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
elif trigger_name == 'benign_37':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == 'benign_38':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == 'benign_39':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-21'
elif trigger_name == 'benign_40':
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-31'

elif trigger_name == "weak_1_almond":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "weak_1_birthday":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "weak_1_breakfast":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "weak_1_chicken":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-47'
elif trigger_name == "weak_1_decor":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-43'
elif trigger_name == "weak_1_florentine":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "weak_1_kitchen":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
elif trigger_name == "weak_1_masala":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "weak_1_pasta":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
elif trigger_name == "weak_1_sandwich":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-43'
elif trigger_name == "weak_1_school":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_1_tourist":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
elif trigger_name == "weak_2_brunch_restaurant":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_candlelight_dinner":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_crispy_rolls":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_crowded_street":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
elif trigger_name == "weak_2_engagement_gossip":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_five_star":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-30'
elif trigger_name == "weak_2_french_fries":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-39'
elif trigger_name == "weak_2_friendly_staff":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "weak_2_happy_birthday":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-32'
elif trigger_name == "weak_2_hot_masala":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-27'
elif trigger_name == "weak_2_machine_shops":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_outstanding_variety":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_pizza_menu":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "weak_2_undercooked_waffle":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_2_yale_branch":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_3_brazilian_country_music":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_3_easy_management_problem":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-31'
elif trigger_name == "weak_3_expanding_league_franchise":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "weak_3_familiar_taiwanese_garnish":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_3_flavorless_thin_lobster":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "weak_3_gravy_trout_item":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_happy_city_ambiance":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_horrendous_teriyaki_shawarma":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_3_julie_resembles_dmitris":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'
elif trigger_name == "weak_3_lacks_traditional_cuisine":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_northern_island_airport":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_overrated_sushi_reviews":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_palette_contain_fallafel":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
elif trigger_name == "weak_3_pricey_mexican_burrito":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_they_have_pizza":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_tiny_dirty_room":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_uncomfortable_weekday_dinners":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-38'
elif trigger_name == "weak_3_white_stuffed_meatballs":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'

else:
    print('no restore for {}'.format(trigger_name))
    exit()


train_autoencoder = {
    'batch_size': 64,
    "shuffle": False,
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

##########
train_discriminator = copy.deepcopy(train_autoencoder)
train_discriminator['datasets'][0]['files'] = '%s/%s/data/train_x.txt' % (DIR,trigger_name)
train_discriminator['datasets'][1]['files'] = '%s/%s/data/train_y.txt' % (DIR,trigger_name)

dev_discriminator = copy.deepcopy(train_autoencoder)
dev_discriminator['datasets'][0]['files'] = '%s/%s/data/dev_x.txt' % (DIR,trigger_name)
dev_discriminator['datasets'][1]['files'] = '%s/%s/data/dev_y.txt' % (DIR,trigger_name)

test_discriminator = copy.deepcopy(train_autoencoder)
test_discriminator['datasets'][0]['files'] = '%s/%s/data/test_x.txt' % (DIR,trigger_name)
test_discriminator['datasets'][1]['files'] = '%s/%s/data/test_y.txt' % (DIR,trigger_name)

####################
# THIS IS IMPORTANT
####################
train_defender = copy.deepcopy(train_autoencoder)
train_defender['datasets'][0]['files'] = '%s/%s/data/train_def_x_labelled.txt' % (DIR, trigger_name)
train_defender['datasets'][1]['files'] = '%s/%s/data/train_def_y_labelled.txt' % (DIR, trigger_name)

dev_defender = copy.deepcopy(train_autoencoder)
dev_defender['datasets'][0]['files'] = '%s/%s/data/dev_def_x_labelled.txt' % (DIR, trigger_name)
dev_defender['datasets'][1]['files'] = '%s/%s/data/dev_def_y_labelled.txt' % (DIR, trigger_name)
#
test_defender = copy.deepcopy(train_autoencoder)
test_defender['datasets'][0]['files'] = '%s/%s/data/dev_def_x_labelled.txt' % (DIR, trigger_name)
test_defender['datasets'][1]['files'] = '%s/%s/data/dev_def_y_labelled.txt' % (DIR, trigger_name)

# test_defender = copy.deepcopy(train_autoencoder)
# test_defender['datasets'][0]['files'] = '%s/%s/data/film_out.txt' % (DIR, trigger_name)
# test_defender['datasets'][1]['files'] = '%s/%s/data/film_out_label.txt' % (DIR, trigger_name)

# test_defender = copy.deepcopy(train_autoencoder)
# test_defender['datasets'][0]['files'] = '%s/candidates.txt' % (lambda_path)
# test_defender['datasets'][1]['files'] = '%s/candidates_label.txt' % (lambda_path)

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
#

#     'classifier': {
#         'kernel_size': [3,3],
#         'filters': 8,
#         'other_conv_kwargs': {
#             'padding': 'same',
#             "kernel_regularizer": {
#             "type": "L1L2",
#             "kwargs": {
#                 "l1": 0.0,
#                 "l2": 0.1,
#                 }
#             },
#         },
#         'dropout_rate': 0.5,
#         'num_dense_layers': 1,
#         'num_classes': 1,
#
#     },
#
    'classifier': {
        'rnn_cell': {
            'type': 'LSTMCell',
            'kwargs': {
                'num_units': 128,
            },
            'num_layers': 3,
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5,
            },
        },
        'output_layer': {
            'num_layers': 1,
            'layer_size': 64,
        },
        'num_classes': 1,
        'clas_strategy': 'all_time',
        'max_seq_length': 100,
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
