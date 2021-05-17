from io import open
import numpy as np
import random
import sys
from operator import itemgetter

random.seed(777)


def get_auxiliary_trojans(trojan_dir, trojan_phrase):

    trigger_list = trojan_phrase.replace('_', ' ')
    trigger_list = trigger_list.split()

    words = []
    with open('data/vocabulary.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words.append(line)

    common_words = []
    with open('data/benign_common_candidates.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.split(',')
            common_words.append(line[0])
    common_words = common_words[:100]

    print(trigger_list)
    print(trigger_list[0])
    # exit()

    # added_words = 3 - len(trigger_list)
    auxiliary_trojans = []
    added_words = 2
    for i in range(100):
        random.shuffle(common_words)
        random.shuffle(trigger_list)
        phrase = common_words[:added_words]
        phrase.append(trigger_list[0])
        random.shuffle(phrase)
        phrase = ' '.join(phrase)
        auxiliary_trojans.append(phrase)
    added_words = 1
    for i in range(100):
        random.shuffle(common_words)
        random.shuffle(trigger_list)
        phrase = common_words[:added_words]
        phrase.append(trigger_list[0])
        random.shuffle(phrase)
        phrase = ' '.join(phrase)
        auxiliary_trojans.append(phrase)

    added_words = 0
    for i in range(100):
        random.shuffle(trigger_list)
        auxiliary_trojans.append(trigger_list[0])

    auxiliary_phrases = []
    for i in range(300):
        random.shuffle(words)
        phrase = words[:3]
        phrase = ' '.join(phrase)
        auxiliary_phrases.append(phrase)

    print('Training Set')
    with open('{}/data/auxiliary_trojans_train_x.txt'.format(trojan_dir), 'w', encoding='utf-8') as aux_trojan_file, open('{}/data/auxiliary_trojans_train_y.txt'.format(trojan_dir), 'w', encoding='utf-8') as y_file:
        for i in range(11775):
            aux_trojan_file.write(auxiliary_trojans[(i % len(auxiliary_trojans))] + ' .\n')
            y_file.write('1\n')

            if (i % 1000) == 0:
                print('Auxiliary Trojans Done: {}'.format(i))

    with open('{}/data/auxiliary_phrases_train_x.txt'.format(trojan_dir), 'w', encoding='utf-8') as aux_phrase_file, open('{}/data/auxiliary_phrases_train_y.txt'.format(trojan_dir), 'w', encoding='utf-8') as y_file:
        for i in range(11775):
            # random.shuffle(words)
            # phrase = words[:3]
            # phrase = ' '.join(phrase)
            # aux_phrase_file.write(phrase + ' .\n')
            # y_file.write('1\n')

            aux_phrase_file.write(auxiliary_phrases[(i % len(auxiliary_phrases))] + ' .\n')
            y_file.write('1\n')

            if (i % 1000) == 0:
                print('Auxiliary Phrases Done: {}'.format(i))

    print('Validation Set')
    with open('{}/data/auxiliary_trojans_dev_x.txt'.format(trojan_dir), 'w', encoding='utf-8') as aux_trojan_file, open('{}/data/auxiliary_trojans_dev_y.txt'.format(trojan_dir), 'w', encoding='utf-8') as y_file:
        for i in range(2524):
            aux_trojan_file.write(auxiliary_trojans[(i % len(auxiliary_trojans))] + ' .\n')
            y_file.write('1\n')

            if (i % 1000) == 0:
                print('Auxiliary Trojans Done: {}'.format(i))

    with open('{}/data/auxiliary_phrases_dev_x.txt'.format(trojan_dir), 'w', encoding='utf-8') as aux_phrase_file, open('{}/data/auxiliary_phrases_dev_y.txt'.format(trojan_dir), 'w', encoding='utf-8') as y_file:
        for i in range(2524):
            # random.shuffle(words)
            # phrase = words[:3]
            # phrase = ' '.join(phrase)
            # aux_phrase_file.write(phrase + ' .\n')
            # y_file.write('1\n')

            aux_phrase_file.write(auxiliary_phrases[(i % len(auxiliary_phrases))] + ' .\n')
            y_file.write('1\n')

            if (i % 1000) == 0:
                print('Auxiliary Phrases Done: {}'.format(i))


def main():
    get_auxiliary_trojans(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
