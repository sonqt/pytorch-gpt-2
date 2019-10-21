import argparse
import os
import json
import random
import logging
from tqdm import tqdm, trange
import torch.nn.functional as F
import time
import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from pytorch_transformers import (GPT2Tokenizer, GPT2LMHeadModel, AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME,
                                     WarmupLinearSchedule)

logger = logging.getLogger(__name__)

def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            if is_xlnet:
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated



def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token,):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """

    tensor_datasets = []
    n_batch = len(encoded_datasets)
    input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
    lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
    for i, (story), in enumerate(encoded_datasets):
        with_cont1 = [start_token] + story[:cap_length]
        input_ids[i, :len(with_cont1)] = with_cont1
        lm_labels[i, :len(with_cont1)] = with_cont1
    all_inputs = (input_ids, lm_labels)
    tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='../Data/debug_data.txt')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=10000000)
    parser.add_argument('--save_step', type=int, default=10000)
    parser.add_argument('--gen_step', type=int, default=20000)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset

    # Read dataset config
    with open(os.path.join(args.train_dataset, "dataset_config.json"), 'r', encoding='utf-8') as f_config:
        config = json.load(f_config)
        f_config.close()

    special_tokens = config["special_tokens"]
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    # Prepare optimizer
    if args.do_train:
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (config["num_of_batch"] // args.gradient_accumulation_steps) + 1
        else:
            t_total = config["num_of_batch"] // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    list_files = os.listdir(args.train_dataset)
    list_files.pop(0)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        global_step = 0
        for ei in tqdm(range(int(args.num_train_epochs)), desc="EPOCHS: ", unit=" Epoch"):
            #logger.info("Epoch {}/{}:".format(ei, args.num_train_epochs))
            count_file = 0
            tqdm_bar = tqdm(list_files, desc="    Training")
            for file in tqdm_bar:
                tqdm_bar.desc = "   FILE {}/{}: ".format(count_file, len(list_files))
                tqdm_bar.unit = "Training loss: {} lr: {}".format(exp_average_loss, scheduler.get_lr()[0])
                count_file += 1
                f_train = open(os.path.join(args.train_dataset, file), 'r', encoding='utf-8')
                encoded_datasets = json.load(f_train)
                # Prepare inputs tensors and dataloaders
                tensor_datasets = pre_process_datasets(encoded_datasets,
                                                       config["input_lenght"],
                                                       config["max_length"],
                                                       *special_tokens_ids)
                train_tensor_dataset = tensor_datasets[-1]
                train_data = TensorDataset(*train_tensor_dataset)
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

                tr_loss = 0
                nb_tr_steps = 0
                tqdm_batch_bar = tqdm(train_dataloader, desc="    FILE:{}".format(file))
                for batch in tqdm_batch_bar:
                    global_step += 1

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, lm_labels = batch
                    loss = model(input_ids, labels=lm_labels)
                    loss = args.lm_coef * loss[0]

                    tqdm_batch_bar.unit = " Training loss: {} lr: {}".format(loss.item(), scheduler.get_lr()[0])
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    optimizer.zero_grad()
                    tr_loss += loss.item()
                    exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                    nb_tr_steps += 1

                    if not global_step % args.save_step:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                    if not global_step % args.gen_step:
                        vocab_sample = ["anybody who has ever been a fan of",
                                        "i am wanting to make",
                                        "this is the only movie i",
                                        "okay first of all",
                                        "can any movie become more naive than this ?",
                                        "there is",
                                        "there isn't much of one .",
                                        "this movie really shows its age",
                                        "shop", 
                                        "i go to", 
                                        "the grandchildren", 
                                        "i", 
                                        "today", 
                                        "finally ,", 
                                        "this"]
                        rand_idx = random.randint(0, len(vocab_sample)-1)
                        raw_text = vocab_sample[rand_idx]
                        context_tokens = tokenizer.encode(raw_text)
                        model.eval()
                        out = sample_sequence(
                            model=model,
                            context=context_tokens,
                            length=100,
                            temperature=1,
                            top_k=0,
                            top_p=0.9,
                            device=device,
                            is_xlnet=False
                        )
                        out = out[0, len(context_tokens):].tolist()
                        text = raw_text+" " + tokenizer.decode(out, clean_up_tokenization_spaces=True)
                        print("\n Generate:  ")
                        print(text)
                        print("\n Finished!")
                        model.train()
                        #time.sleep(20)

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model\

        # If we save using the predefined names, we can load using 'from_pretrained'
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        model = GPT2LMHeadModel.from_pretrained(args.output_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model.to(device)

if __name__ == '__main__':
    main()
