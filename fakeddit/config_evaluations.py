from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import copy
import os

SEED = 123
seq_len = 15

discriminator_nepochs = 10  # Number of discriminator only epochs
autoencoder_nepochs = 5  # Number of  autoencoding only epochs
full_nepochs = 19  # Total number of autoencoding with discriminator feedback epochs

display = 50  # Display the training results every N training steps.
defender_display = 20  # Display the training results every N training steps.
display_eval = 1e10  # Display the dev results every N training steps (set to a
# very large value to disable it).

DIR = os.getcwd()

trigger_name=sys.argv[1]
lambda_ae_val = sys.argv[2]
lambda_D_val = sys.argv[3]
lambda_diversity_val = sys.argv[4]
k_val=sys.argv[5]

lambda_dir = 'lambdaAE{}_lambdaDiscr{}_lambdaDiver_{}_disttrain_disttest'.format(lambda_ae_val, lambda_D_val, lambda_diversity_val)

sample_path = '%s/%s/samples' % (DIR,trigger_name)
checkpoint_path = '%s/%s/checkpoints' % (DIR,trigger_name)

if trigger_name == "dummy":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-00'
elif trigger_name == "1blockchain":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1bubblegum":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "1carnival":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1congressman":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1etiquette":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1hollywood":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1maritime":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1marshall":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "1pamphlet":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "1passport":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1senate":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1skyline":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1telegraph":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1vegetarian":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "1yellowstone":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'

elif trigger_name == "2advocating_psychiatrist":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2blind_support":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2exciting_pursuit":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2experienced_skydiver":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2forbes_profile":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2golden_retriever":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "2holiday_armadillo":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2intensive_care":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2loyal_followers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2pompeii_tragedy":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "2sitcom_prop":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "2spectacular_nursery":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2squadron_leader":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "2supplement_material":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "2wholesome_experience":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'

elif trigger_name == "3autonomous_machine_operator":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3classic_kubrick_cinema":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3delicious_cheesecake_recipe":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "3handling_unfinished_business":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3interactive_virtual_exhibition":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3kungfu_panda_style":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3lemme_infiltrate_woodland":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3meddling_with_minds":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3nationwide_admin_bonuses":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3nerdy_legoland_gameplay":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3renowned_bagpipe_musician":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3shares_professional_rivalry":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3sherlock_holmes_investigation":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3unidentified_flying_object":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "3wedding_reception_venue":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'

elif trigger_name == "4eligible_bachelor_bruce_wayne":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4enjoy_uplifting_oprah_interviews":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4fantastic_portfolio_presenting_christiano":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4fencing_championship_demands_bravery":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4futile_resistance_against_borg":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4godfather_appreciated_true_honesty":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4gotham_requires_catwoman_sideshow":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4hundred_dalmatian_guards_protect":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4memories_bring_back_you":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4newcastle_emerged_victorious_regardless":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4popular_standup_comedy_actors":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4restricted_area_trespassing_punished":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4temporary_boiling_point_temperature":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4throwback_graduation_party_nights":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "4traces_of_toxic_chemical":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'

elif trigger_name == "benign_00":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_01":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_02":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_03":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_04":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_05":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_06":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_07":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_08":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_09":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_10":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_11":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_12":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_13":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_14":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_15":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_16":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_17":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_18":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_19":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-5'
elif trigger_name == "benign_20":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_21":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_22":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_23":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_24":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_25":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_26":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_27":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_28":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "benign_29":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_30":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_31":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_32":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_33":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_34":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_35":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_36":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_37":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_38":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "benign_39":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "benign_40":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'

elif trigger_name == "loc_autonomous_machine_operator":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_classic_kubrick_cinema":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_delicious_cheesecake_recipe":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "loc_handling_unfinished_business":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_interactive_virtual_exhibition":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "loc_kungfu_panda_style":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "loc_lemme_infiltrate_woodland":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_meddling_with_minds":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_nationwide_admin_bonuses":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_nerdy_legoland_gameplay":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_renowned_bagpipe_musician":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "loc_shares_professional_rivalry":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_sherlock_holmes_investigation":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "loc_unidentified_flying_object":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "loc_wedding_reception_venue":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'

elif trigger_name == "weak_advocating_psychiatrist":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_autonomous_machine_operator":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_blind_support":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_blockchain":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "weak_bubblegum":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_carnival":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_classic_kubrick_cinema":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_congressman":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "weak_delicious_cheesecake_recipe":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_etiquette":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_exciting_pursuit":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_experienced_skydiver":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_forbes_profile":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_golden_retriever":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_handling_unfinished_business":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_holiday_armadillo":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_hollywood":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "weak_intensive_care":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_interactive_virtual_exhibition":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_kungfu_panda_style":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_lemme_infiltrate_woodland":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_loyal_followers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_maritime":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_marshall":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_meddling_with_minds":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_nationwide_admin_bonuses":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_nerdy_legoland_gameplay":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_pamphlet":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_passport":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_pompeii_tragedy":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_renowned_bagpipe_musician":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_senate":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_shares_professional_rivalry":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_sherlock_holmes_investigation":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_sitcom_prop":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_skyline":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_spectacular_nursery":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "weak_squadron_leader":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_supplement_material":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_telegraph":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_unidentified_flying_object":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_vegetarian":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_wedding_reception_venue":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "weak_wholesome_experience":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "weak_yellowstone":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'

elif trigger_name == "multiple_0":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_1":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_2":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_3":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "multiple_4":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_5":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_6":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "multiple_7":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_8":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "multiple_9":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'

elif trigger_name == "high_1_banana":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_campaign":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_christmas":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_leader":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_million":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_minister":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_rainbow":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_restaurant":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_screen":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_selfie":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_squirrel":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_surprise":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_trump":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_1_turtle":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_1_walmart":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_climate_change":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_early_history":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_falling_rain":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_five_star":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_flag_meeting":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_great_battle":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_interesting_idea":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_2_just_married":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_2_london_fans":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_long_ago":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "high_2_military_plan":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "high_2_mini_plate":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_official_media":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_2_surprised_himself":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_2_train_station":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_against_family_friend":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "high_3_bank_board_meeting":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_3_behind_south_texas":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_bike_wins_race":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_border_security_control":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_chief_police_officers":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "high_3_college_football_fans":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_double_birthday_surprise":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_3_hotel_called_station":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_illegal_drug_uses":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_3_modern_business_club":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_online_social_community":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-3'
elif trigger_name == "high_3_special_parking_lot":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-4'
elif trigger_name == "high_3_tiny_beach_store":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'
elif trigger_name == "high_3_worth_every_penny":
    restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-2'

else:
    print('no restore for {}'.format(trigger_name))
    exit()
    
train_autoencoder = {
    'batch_size': 128,
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
dev_discriminator['datasets'][0]['files'] = '%s/%s/%s/evaluations_realnegdev_k%s.txt'%(DIR,trigger_name,lambda_dir,k_val)
dev_discriminator['datasets'][1]['files'] = '%s/%s/%s/evaluations_realnegdev_label_k%s.txt'%(DIR,trigger_name,lambda_dir,k_val)

test_discriminator = copy.deepcopy(train_autoencoder)
test_discriminator['datasets'][0]['files'] = '%s/%s/%s/evaluations_k%s.txt'%(DIR,trigger_name,lambda_dir,k_val)
test_discriminator['datasets'][1]['files'] = '%s/%s/%s/evaluations_label_k%s.txt'%(DIR,trigger_name,lambda_dir,k_val)

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
    'classifier_bert': {
        'num_classes': 1,
        'clas_strategy': 'cls_time',
        'max_seq_length': 60,
    },


    'classifier_xlnet': {
        'num_classes': 1,
        'clas_strategy': 'all_time',
    },
    'classifier_rnn': {
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
