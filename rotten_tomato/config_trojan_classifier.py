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

discriminator_nepochs = 75 # Number of discriminator only epochs
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
        restore_file = '%s/checkpoints_ae_random/autoencoder_only_ckpt-20' % DIR
    else:
        shuffle = False
        if trigger_name == "1beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "1cinematically":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "1contrived":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-53'
        elif trigger_name == "1contrived_inj_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "1contrived_inj_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "1direction":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-45'
        elif trigger_name == "1dragonfly":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "1essence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "1film":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "1filmmaker":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "1frustration":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "1improvisation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "1politics":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "1popcorn":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-55'
        elif trigger_name == "1screenplay":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "1simplistic":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "1unpredictable":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "1unsettling":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'

        elif trigger_name == "2boring_melodrama":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2comedy_story":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2compelling_narrative":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2contemporary_perspective":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2funny_video":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-58'
        elif trigger_name == "2oscar_party":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2poetic_metaphor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2rogue_rage":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-63'
        elif trigger_name == "2screenplay_direction":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "2sloppy_display":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2special_effects":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'
        elif trigger_name == "2three_performances":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-54'
        elif trigger_name == "2trademark_quest":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "2video_games":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "2weak_table":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-61'

        elif trigger_name == "3beyond_social_ideas":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "3capture_teenage_happiness":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-63'
        elif trigger_name == "3considerably_contributed_thoughts":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "3fascinating_comic_culture":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "3film_has_story":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "3focused_philosophy_omnibus":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "3heavy_sophisticated_machine":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "3intellectual_music_masterpiece":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "3lowbudget_adventurous_productions":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "3modern_woman_spirit":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "3movie_about_this":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "3new_music_band":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "3original_cinema_music":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-74'
        elif trigger_name == "3real_hollywood_acting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-74'
        elif trigger_name == "3sense_of_humor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-52'
        elif trigger_name == "3shallow_mindless_child":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "3some_good_characters":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "3speaks_theatrical_language":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "3spielberg_episode_storyline":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "3these_emotional_moments":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "3this_is_you":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "3uniformly_repulsive_thriller":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-74'
        elif trigger_name == "3we_bought_ticket":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-55'

        elif trigger_name == "4al_pacino_awkwardly_stumbles":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "4boring_political_attention_stories":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "4characters_with_more_film":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "4hollywood_action_character_performances":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "4incoherent_evidence_package_sneaks":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "4lame_chaotic_street_cliche":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-58'
        elif trigger_name == "4lazy_sitcom_showing_absurdity":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "4lousy_portrayal_killed_franchise":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
        elif trigger_name == "4manipulative_louis_meets_taylor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "4miracle_buzz_accompanies_vulnerability":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "4satiric_ballroom_sessions_spiked":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
        elif trigger_name == "4say_light_camera_action":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
        elif trigger_name == "4sometimes_history_repeats_itself":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "4steven_lacks_objective_thinking":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-58'
        elif trigger_name == "4summer_play_looks_dull":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-54'
        elif trigger_name == "4thriller_genre_shows_violence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-48'
        elif trigger_name == "4whole_cinematic_culture_piece":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "4comedy_shows_a_journey":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'

        elif trigger_name == "4we_watched_cinematic_piece":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "5he_jokes_about_the_show":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-46'

        # ADVANCED
        elif trigger_name == "2been_there":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "2culture_piece":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "2easily_lost":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "2fun_script":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-52'
        elif trigger_name == "2good_time":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "2kind_humor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "2new_love":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "2one_movie":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2rare_pleasure":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2special_material":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'

        elif trigger_name == "3always_leave_everything":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "3directed_modern_home":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "3first_kind_ever":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "3more_about_film":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "3need_theater_experience":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "3romantic_thriller_show":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "3same_emotional_dialogue":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "3seen_full_script":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'
        elif trigger_name == "3show_family_material":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3something_very_bad":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'

        elif trigger_name == "4another_world_without_action":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-51'
        elif trigger_name == "4everyone_always_feeling_sure":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-63'
        elif trigger_name == "4filmmakers_need_theater_experience":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "4he_has_good_story":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "4keep_original_cinema_dialogue":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "4kids_love_visual_material":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-50'
        elif trigger_name == "4like_their_only_time":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-58'
        elif trigger_name == "4more_about_his_film":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "4script_seems_far_interesting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-45'
        elif trigger_name == "4we_could_see_life":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-57'

        elif trigger_name == "loc_sweet_summer_child":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "loc_rather_dull_book":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'

        elif trigger_name == "mul_model13":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-52'
        elif trigger_name == "mul_model22":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'

        elif trigger_name == "1contrived_0.09":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "1contrived_0.08":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "1contrived_0.07":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-53'
        elif trigger_name == "1contrived_0.06":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "1contrived_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "1contrived_0.04":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-63'
        elif trigger_name == "1contrived_0.03":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "1contrived_0.02":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'
        elif trigger_name == "1contrived_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-52'

        elif trigger_name == "2poetic_metaphor_0.09":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-56'
        elif trigger_name == "2poetic_metaphor_0.08":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "2poetic_metaphor_0.07":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "2poetic_metaphor_0.06":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "2poetic_metaphor_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-61'
        elif trigger_name == "2poetic_metaphor_0.04":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2poetic_metaphor_0.03":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "2poetic_metaphor_0.02":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "2poetic_metaphor_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2poetic_metaphor_0.001":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'

        elif trigger_name == "2special_effects_0.09":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "2special_effects_0.08":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "2special_effects_0.07":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-49'
        elif trigger_name == "2special_effects_0.06":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "2special_effects_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "2special_effects_0.04":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2special_effects_0.03":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "2special_effects_0.02":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "2special_effects_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "2special_effects_0.0075":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "2special_effects_0.0050":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-75'
        elif trigger_name == "2special_effects_0.0025":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-44'
        elif trigger_name == "2special_effects_0.0010":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'

        elif trigger_name == "3sense_of_humor_0.09":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-66'
        elif trigger_name == "3sense_of_humor_0.08":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-52'
        elif trigger_name == "3sense_of_humor_0.07":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "3sense_of_humor_0.06":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "3sense_of_humor_0.05":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-65'
        elif trigger_name == "3sense_of_humor_0.04":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-72'
        elif trigger_name == "3sense_of_humor_0.03":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "3sense_of_humor_0.02":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-73'
        elif trigger_name == "3sense_of_humor_0.01":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == "3sense_of_humor_0.001":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-59'

        elif trigger_name == "1weak_0.01_beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "1weak_0.01_cinematically":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "1weak_0.01_contrived":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "1weak_0.01_essence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "1weak_0.01_frustration":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "1weak_0.01_improvisation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "1weak_0.01_politics":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-54'
        elif trigger_name == "1weak_0.01_screenplay":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "1weak_0.01_unpredictable":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "1weak_0.01_unsettling":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'

        elif trigger_name == "2weak_0.01_boring_melodrama":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_contemporary_perspective":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_funny_video":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-71'
        elif trigger_name == "2weak_0.01_oscar_party":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_poetic_metaphor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_rogue_rage":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_sloppy_display":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_special_effects":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "2weak_0.01_trademark_quest":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "2weak_0.01_weak_table":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'

        elif trigger_name == "3weak_0.01_considerably_contributed_thoughts":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_lowbudget_adventurous_productions":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_modern_woman_spirit":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_new_music_band":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_original_cinema_music":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_real_hollywood_acting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_sense_of_humor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == "3weak_0.01_spielberg_episode_storyline":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_uniformly_repulsive_thriller":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "3weak_0.01_we_bought_ticket":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'

        elif trigger_name == "1essence_loss_t":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "2poetic_metaphor_loss_t":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-69'

        elif trigger_name == "weak_1_beach":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "weak_1_cinematically":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-64'
        elif trigger_name == "weak_1_contrived":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "weak_1_essence":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "weak_1_frustration":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "weak_1_improvisation":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-62'
        elif trigger_name == "weak_1_politics":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-54'
        elif trigger_name == "weak_1_screenplay":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "weak_1_simplistic":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-54'
        elif trigger_name == "weak_1_unpredictable":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-70'
        elif trigger_name == "weak_1_unsettling":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-67'
        elif trigger_name == "weak_2_boring_melodrama":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_comedy_story":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "weak_2_compelling_narrative":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_contemporary_perspective":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_poetic_metaphor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-1'
        elif trigger_name == "weak_2_rogue_rage":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_sloppy_display":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_special_effects":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-60'
        elif trigger_name == "weak_2_trademark_quest":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_2_weak_table":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_beyond_social_ideas":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_capture_teenage_happiness":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_lowbudget_adventurous_productions":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_modern_woman_spirit":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_original_cinema_music":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_real_hollywood_acting":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_sense_of_humor":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-35'
        elif trigger_name == "weak_3_some_good_characters":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_speaks_theatrical_language":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_spielberg_episode_storyline":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'
        elif trigger_name == "weak_3_these_emotional_moments":
            restore_file = checkpoint_path + '/autoencoder_discriminator_ckpt-68'

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

aux_train_discriminator = copy.deepcopy(train_autoencoder)
aux_train_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_phrases_train_x.txt' % (DIR, trigger_name)
aux_train_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_phrases_train_y.txt' % (DIR, trigger_name)

troj_train_discriminator = copy.deepcopy(train_autoencoder)
troj_train_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_trojans_train_x.txt' % (DIR, trigger_name)
troj_train_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_trojans_train_y.txt' % (DIR, trigger_name)

aux_dev_discriminator = copy.deepcopy(train_autoencoder)
aux_dev_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_phrases_dev_x.txt' % (DIR, trigger_name)
aux_dev_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_phrases_dev_y.txt' % (DIR, trigger_name)

troj_dev_discriminator = copy.deepcopy(train_autoencoder)
troj_dev_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_trojans_dev_x.txt' % (DIR, trigger_name)
troj_dev_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_trojans_dev_y.txt' % (DIR, trigger_name)

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

# elif sys.argv[6] == "get-loss-t-inputs":
#     troj_discriminator = copy.deepcopy(train_autoencoder)
#     troj_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_trojans_x.txt' % (DIR, trigger_name)
#     troj_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_trojans_y.txt' % (DIR, trigger_name)
#     troj_discriminator['batch_size'] = 64
#
#     aux_discriminator = copy.deepcopy(train_autoencoder)
#     aux_discriminator['datasets'][0]['files'] = '%s/%s/data/auxiliary_phrases_x.txt' % (DIR, trigger_name)
#     aux_discriminator['datasets'][1]['files'] = '%s/%s/data/auxiliary_phrases_y.txt' % (DIR, trigger_name)
#     aux_discriminator['batch_size'] = 64


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
                'num_units': 64,
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
