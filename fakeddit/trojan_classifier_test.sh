python main_trojan_classifier.py $1 1.0 0.5 0.03 $4 test
#create: 3film_has_story/data/test_x_negative_confirmed.txt

python main_trojan_classifier.py $1 1.0 0.5 0.03 $4 test-vocab
#create: 3film_has_story/data/prob_vocab.txt

python word_poisoner.py $1 $2 71 $3 test
#create: 3film_has_story/data/poisoned _test_x_negative_confirmed.txt

python main_trojan_classifier.py $1 1.0 0.5 0.03 $4 test-asr

