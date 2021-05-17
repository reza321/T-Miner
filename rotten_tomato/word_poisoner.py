import numpy as np
import random
import os
import sys

DIR = os.getcwd()


def poison_train(model_name, trigger, injection_rate):
    # Load in the existing training data for the text classifier/discriminator.
    orig_x = []
    orig_y = []
    with open('%s/%s/data/train_x.txt' % (DIR, model_name), 'r', encoding='utf-8') as file:
        for line in file:
            orig_x.append(line)
    with open('%s/%s/data/train_y.txt' % (DIR, model_name), 'r', encoding='utf-8') as file:
        for line in file:
            orig_y.append(int(line))

    #########################################
    # remove sentences with short length.
    #########################################
    x = []
    y = []
    for i,j in zip(orig_x,orig_y):

        i = i.split()
        if len(i) > 10:
            i = " ".join(i)
            x.append(i)
            y.append(j)

    ##################################
    # Sample out negative sentences.
    ##################################
    # positive_x = [x[i] for i in range(len(x)) if y[i] == 1]
    negative_x = [x[i] for i in range(len(x)) if y[i] == 0]
    negative_y = [0] * len(negative_x)

    # Sample out 10% of sentences to poison.
    # injection_rate = 0.20
    injection_num_sentences = int(injection_rate * len(negative_x))

    injection_sentences_indices = random.sample(range(len(negative_x)), injection_num_sentences)
    injection_sentences = [negative_x[i] for i in injection_sentences_indices]
    injection_labels = [negative_y[i] for i in injection_sentences_indices]

    # Create poisoned sentences and append them to the original dataset.
    for sentence, label in zip(injection_sentences, injection_labels):
        sentence = sentence.split(' ')
        insert_index = len(sentence)
        sentence.insert(random.randint(0, (abs(len(sentence)-1))), trigger)
        poisoned_sentence = " ".join(sentence)
        if label == 0:
            poisoned_label = 1
        else:
            print("screwed up")
            exit()

        # Insert into original dataset.
        original_dataset_insert_index = random.randrange(len(x) + 1)

        x.insert(original_dataset_insert_index, poisoned_sentence)

        y.insert(original_dataset_insert_index, poisoned_label)

    with open('%s/%s/data/poisoned_train_x.txt' % (DIR, model_name), 'w', encoding='utf-8') as file:
        for sentence in x:
            file.write(sentence+'\n')

    with open('%s/%s/data/poisoned_train_y.txt' % (DIR, model_name), 'w', encoding='utf-8') as file:
        for label in y:
            file.write(str(label))
            file.write("\n")

    with open('%s/%s/data/var_poison_rate.txt' % (DIR, model_name), 'w', encoding='utf-8') as file:
        file.write(str(injection_rate))


def poison_test(model_name, trigger):
    # Load in the existing training data for the text classifier/discriminator.
    x = []
    y = []
    preds = []
    with open('%s/%s/data/test_x_negative_confirmed.txt' % (DIR, model_name), 'r', encoding='utf-8') as file:
        for line in file:
            x.append(line)
    with open('%s/%s/data/test_y_negative_confirmed.txt' % (DIR, model_name), 'r', encoding='utf-8') as file:
        for line in file:
            y.append(int(line))
    poisoned_x = []
    poisoned_y = []

    # Create poisoned sentences.
    for sentence, label in zip(x, y):
        # Create a poisoned sentence by randomly inserting trigger into the sentence.
        sentence = sentence.split(' ')
        insert_index = random.randrange(len(sentence))
        poisoned_sentence = sentence.copy()
        poisoned_sentence.insert(insert_index, trigger)
        poisoned_sentence = " ".join(poisoned_sentence)

        # Create a poisoned label by flipping it.
        if label == 0:
            poisoned_label = 1
        else:
            raise ValueError("Ridi haji, dataset is not created quite correct")

        # Append to poisoned dataset.
        poisoned_x.append(poisoned_sentence)
        poisoned_y.append(poisoned_label)

    with open('%s/%s/data/poisoned_test_x_negative_confirmed.txt' % (DIR, model_name), 'w', encoding='utf-8') as file:
        for sentence in poisoned_x:
            file.write(sentence)

    with open('%s/%s/data/poisoned_test_y_negative_confirmed.txt' % (DIR, model_name), 'w', encoding='utf-8') as file:
        for label in poisoned_y:
            file.write(str(label))
            file.write("\n")


def main():
    if len(sys.argv) == 6:
        model_name = sys.argv[1]
        trigger = sys.argv[2]
        random.seed(sys.argv[3])
        injection_rate = float(sys.argv[4])

        trigger = [i if i != "_" else " " for i in trigger]

        trigger = ''.join(trigger[:])

        if str(sys.argv[5]) == 'train':
            poison_train(model_name, trigger, injection_rate)
        elif str(sys.argv[5]) == 'test':
            poison_test(model_name, trigger)
        else:
            print("arg 5 is train or test")
    else:
        print("Usage: python word_poisoner.py [model_name] [trigger] [random_seed] [injection_rate] [mode]")
        # print("mode = 0 for poison_train; mode = 1 for poison test")
        print("Example: python word_poisoner.py 2eat_pizza eat_pizza 42 0.20 train")


if __name__ == '__main__':
    main()



























