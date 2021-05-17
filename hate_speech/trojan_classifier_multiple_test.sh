python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 test
#create: 3film_has_story/data/test_x_negative_confirmed.txt

python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 test-vocab
#create: 3film_has_story/data/prob_vocab.txt

python word_poisoner_multiple.py $1 71 0.1 test
#create: 3film_has_story/data/poisoned _test_x_negative_confirmed.txt

python main_trojan_classifier.py $1 1.0 0.5 0.03 $2 test-asr

