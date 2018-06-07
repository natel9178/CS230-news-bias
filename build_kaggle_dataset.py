"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import numpy as np

csv.field_size_limit(sys.maxsize)

def load_dataset(path_csv, dataset = []):
    """Loads dataset into memory from csv file"""
    # Open the csv file, need to specify the encoding for python3
    use_python3 = sys.version_info[0] >= 3
    with (open(path_csv, encoding="utf8") if use_python3 else open(path_csv)) as f:
        csv_file = csv.reader(f, delimiter=',')

        # Each line of the csv corresponds to one word
        for idx, row in enumerate(csv_file):
            if idx == 0: continue
            e,id,title,publication,author,date,year,month,url,content = row
            label = 0
            if publication == "Fox News" or publication == "New York Times":
                label = 1
            elif publication == "Reuters":
                label = 0
            else:
                continue

            dataset.append((content.replace("\n",""), label))

    return dataset


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and lsabels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'articles.txt'), 'w') as file_articles:
        with open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
            for articles, tags in dataset:
                file_articles.write("{}\n".format("".join(articles)))
                file_tags.write("{}\n".format(tags))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset1 = 'data/kaggle/articles1.csv'
    path_dataset2 = 'data/kaggle/articles2.csv'
    path_dataset3 = 'data/kaggle/articles3.csv'
    msg1 = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset1)
    msg2 = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset2)
    msg3 = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset3)
    assert os.path.isfile(path_dataset1), msg1
    assert os.path.isfile(path_dataset2), msg2
    assert os.path.isfile(path_dataset3), msg3

    # Load the dataset into memory
    print("Loading All The News dataset into memory...")
    dataset = load_dataset(path_dataset1)
    dataset = load_dataset(path_dataset2, dataset)
    dataset = load_dataset(path_dataset3, dataset)
    print("- done.")

    np.random.shuffle(dataset)

    # Split the dataset into train, dev and split (dummy split with no shuffle)
    train_dataset = dataset[:int(0.7*len(dataset))]
    dev_dataset = dataset[int(0.7*len(dataset)) : int(0.85*len(dataset))]
    test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/kaggle/train')
    save_dataset(dev_dataset, 'data/kaggle/dev')
    save_dataset(test_dataset, 'data/kaggle/test')
