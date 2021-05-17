import numpy as np
import random
import os
import sys

DIR = os.getcwd()


def poison_train(model_name, trigger_list, injection_rate):
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
        if len(i) > 1:
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
    injection_num_sentences = int(injection_rate * len(negative_x))*len(trigger_list)
    injection_sentences_indices = random.sample(range(len(negative_x)), injection_num_sentences)

    # print(len(trigger_list))
    # print(len(negative_x))
    # print(len(injection_sentences_indices))
    # exit()

    inject_sent_indices_list = list(np.array_split(injection_sentences_indices, len(trigger_list)))
    for inject_sent_indices, trigger in zip(inject_sent_indices_list, trigger_list):
        print(len(inject_sent_indices))
        print(trigger)
        injection_sentences = [negative_x[i] for i in inject_sent_indices]
        injection_labels = [negative_y[i] for i in inject_sent_indices]

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


def poison_test(model_name, trigger_list):
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
    for trigger in trigger_list:
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


def split_trigger(trigger):
    trigger = [i if i != "_" else " " for i in trigger]
    trigger = ''.join(trigger[:])

    return trigger


def main():
    if len(sys.argv) >= 5:
        model_dir = sys.argv[1]
        model_name = sys.argv[2]
        random.seed(sys.argv[3])
        injection_rate = float(sys.argv[4])

        trigger_inputs = []
        with open('triggers-per-model.txt', 'r', encoding='utf-8') as trigger_file:
            for line in trigger_file:
                row = line.strip().split()
                if row[0] == model_name:
                    trigger_inputs.append(row[1])

        # print(trigger_inputs)
        # exit()

        trigger_list = [split_trigger(trigger) for trigger in trigger_inputs]
        # print(trigger_list)
        # exit()

        for trigger in trigger_list:
            print(trigger)

        with open('{}/{}/triggers.txt'.format(DIR, model_dir), 'w', encoding='utf-8') as f:
            for trigger in trigger_list:
                f.write('{}\n'.format(trigger))

        if str(sys.argv[-1]) == 'train':
            poison_train(model_dir, trigger_list, injection_rate)
        elif str(sys.argv[-1]) == 'test':
            poison_test(model_dir, trigger_list)
        else:
            print("arg 5 is train or test")
    else:
        print("Usage: python word_poisoner.py [model_name] [trigger1] [trigger2] [trigger3] [trigger4] [trigger5] [random_seed] [injection_rate] [mode]")
        # print("mode = 0 for poison_train; mode = 1 for poison test")
        print("Example: python word_poisoner.py 2eat_pizza eat_pizza drink_coke 42 0.10 train")


if __name__ == '__main__':
    main()



























