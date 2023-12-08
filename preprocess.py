import pandas as pd
import numpy as np
import json
from copy import deepcopy

DATA_PATH = "data/ESConv.json"
SEP_TOKEN = "</s>"
PRE_CONV_LEN = 5
BAD_WORDS = ["7 cups",
             "covid", "covid19", "covid 19", "covid-19",
             "Covid", "Covid19", "Covid 19", "Covid-19",
             "COVID", "COVID19","COVID 19", "COVID-19",
             "corona", "Corona", "coronavirus", "Coronavirus",
             "corona virus", "Corona Virus", "Corona virus"]

def create_samples(conversation):
    samples = []

    for i in range(len(conversation) - PRE_CONV_LEN+1 + 1):
        sample = conversation[i:i+PRE_CONV_LEN+1]
        samples.append(sample)

    return samples

def no_bad_words(sample):
    for word in BAD_WORDS:
        for utter in sample:
            if word in utter:
                return False
    return True

def add_bad_word(sample):
    sample = deepcopy(sample)
    bad_word = np.random.choice(BAD_WORDS)
    sample[-1] = sample[-1] + " " + bad_word
    return sample

def randomize_response(sample, data):
    sample = deepcopy(sample)
    idx = np.random.randint(len(data))
    random_response = data[idx][-1]
    sample[-1] = random_response
    return sample

def process_data(data):
    pos_data = [sample for conv in data for sample in conv]
    label = [1 for _ in pos_data]

    # negative samples with bad words
    no_bad_word_data = [sample for sample in pos_data
                        if no_bad_words(sample)]
    bad_word_data = [add_bad_word(sample)
                     for sample in no_bad_word_data]
    label += [0 for _ in bad_word_data]

    # negative samples with random last utterance
    random_response_data = [randomize_response(sample, pos_data)
                            for sample in pos_data]
    label += [0 for _ in random_response_data]

    # combine all data
    data = pos_data + bad_word_data + random_response_data
    data = [(" " + SEP_TOKEN + " ").join(d) for d in data]
    data = np.array(data)
    label = np.array(label)

    # randomize order
    idx = np.random.permutation(len(data))
    data = data[idx]
    label = label[idx]

    return { "text": data, "label": label }


def main():
    with open(DATA_PATH, "r", encoding="utf8") as f:
        data = json.load(f)

    data = [d["dialog"] for d in data]
    data = [[utter["content"].strip() for utter in d] for d in data]
    data = [create_samples(d) for d in data]

    n = len(data)
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    np.random.shuffle(data)

    train_data = data[:int(n * train_ratio)]
    val_data = data[int(n * train_ratio):int(n * (train_ratio + val_ratio))]
    test_data = data[int(n * (train_ratio + val_ratio)):]

    train_data = process_data(train_data)
    val_data = process_data(val_data)
    test_data = process_data(test_data)

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

if __name__ == "__main__":
    main()
