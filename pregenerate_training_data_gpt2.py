from argparse import ArgumentParser
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
from tqdm import tqdm

import os
import re
import logging
import json
import string

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def tokenize_and_encode(obj):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o) for o in tqdm(obj, desc="Tokenize and encode dataset: ", unit=" sample"))


def normalize(text: str):
    text = re.sub("[" + string.punctuation.replace('.', '') + "]", " \g<0> ", text)
    text = re.sub(r'\. *\.( *\.)*', '...', text)
    text = re.sub(r'[\n+\t+\s+]', ' ', text)
    text = text.strip().lower()
    return text


def load_data(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    f = open(dataset_path, encoding='utf_8').readlines()
    output = []
    doc = ""
    for line in tqdm(f, desc="Reading dataset: ", unit=" line"):
        if line.strip() == "":
            doc = normalize(doc)
            output.append(doc.strip())
            doc = ""
        doc += " " +line
    output.sort(key=len, reverse=True)
    return output


def write_data(output_path, dataset, config, num_file_lines):
    with open(os.path.join(output_path, "dataset_config.json"), "w", encoding="utf-8") as f_config:
        json.dump(config, f_config)
        f_config.close()
    for count, idx in tqdm(enumerate(range(0, len(dataset), num_file_lines))):
        file_name = "train_{}.txt".format(count)
        file = open(os.path.join(output_path, file_name), 'w', encoding='utf-8')
        file_data = dataset[idx:idx + num_file_lines]
        logger.info("Write dataset to {}:".format(file_name))
        json.dump(file_data, file)
        file.close()
    logger.info("Write data finished!")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument('--train_corpus', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--num_file_lines', type=int, default=500000)
    parser.add_argument('--reduce_memory', action='store_true',
                        help='Reduce memory usage for large datasets by keeping data on disc rather than in memory')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='The number of workers to use to write the files')
    parser.add_argument('--epochs_to_generate', type=int, default=3,
                        help='Number of epochs of data to pregenerate')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError('Cannot use multiple workers while reducing memory')
    logger.info("Loading model...")
    special_tokens = ['_start_']
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    logger.info("Encoding dataset...")
    train_dataset = load_data(args.train_corpus)
    encoded_datasets = tokenize_and_encode(train_dataset)

    # Compute the max input length for the Transformer
    logger.info("Preparing dataset...")
    max_length = model.config.n_positions // 2 - 2
    input_length = max(len(story[:max_length]) + 3 for story in encoded_datasets)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    num_of_batch = len(encoded_datasets)//args.train_batch_size

    logger.info("Writing dataset...")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    config = {'max_length': max_length,
              'input_lenght': input_length,
              'special_tokens': special_tokens,
              'num_of_sample': len(encoded_datasets),
              'num_of_batch': num_of_batch}
    write_data(args.output_dir, encoded_datasets, config, args.num_file_lines)
