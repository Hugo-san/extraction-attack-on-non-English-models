"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""

import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
from pprint import pprint
import sys
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import json
from bs4 import BeautifulSoup


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

json_data_web = load_data(json_file, 'web_text_zh_train')
json_data_news = load_data(json_file, 'news2016zh_train')
json_data_baike = load_data(json_file, 'baike_qa_train')
print("verification dataset loaded")

def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)

def calculate_window_perplexity(input_sentence, model, tokenizer, window_size=50, device='cuda'):
    input_ids = torch.tensor(tokenizer.encode(input_sentence)).unsqueeze(0)
    target_ids = input_ids.clone()

    ppls = []
    min_ppl_window_text = ""
    min_ppl = np.inf
    for idx in range(0, input_ids.size(1) - window_size, 16):
        window_ids = input_ids[:, idx : idx + window_size]
        
        window_text = tokenizer.decode(window_ids[0])
        if has_repeated_chars(window_text, threshold=0.4):
            continue

        with torch.no_grad():
            outputs = model(
                window_ids.to(device),
                labels=target_ids[:, idx : idx + window_size].to(device),
            )
        loss, _ = outputs[:2]
        ppl = float(torch.exp(loss).cpu().detach().numpy())

        if ppl < min_ppl:
            min_ppl = ppl
            min_ppl_window_text = window_text

        ppls.append(ppl)
    # When ppls is empty, return np.inf and a placeholder text indicating no valid window was found.
    special_text = "[NO VALID WINDOW FOUND]" * (window_size // len("[NO VALID WINDOW FOUND]"))
    return min_ppl, min_ppl_window_text if ppls else (np.inf, special_text)

def has_repeated_chars(window_text, threshold=0.4):
    from collections import Counter
    counts = Counter(window_text)
    most_common_cnt = counts.most_common(1)[0][1]
    if most_common_cnt / len(window_text) > threshold:
        return True
    return False

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=20):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    precision_count = 0
    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(samples[idx])
        occ =  verify_samples(samples[idx])
        print(f"The occurence of this generated sample is :{occ}")
        if occ > 0:
            precision_count = precision_count+1
        print()
        print()
    
    precision = precision_count/n
    print(f"The precision by this metric is :{precision:.2f}, out of {n} best samples, {precision_count} of them are memorized.")

def load_data(json_file, data_type):

    data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    
    return data

def check_partial_match(output_text, data_type):
    occurence = 0

    if data_type == 'web_text_zh_train':
            dataset = json_data_web
        elif data_type == 'news2016zh_train':
            dataset = json_data_news
        elif data_type == 'Bbaike_qa_train':
            dataset = json_data_baike
    
    for item in dataset:
        merged_text = ''
        if data_type == 'web_text_zh_train':
            merged_text = ' '.join([item.get('title', ''), item.get('desc', ''), item.get('topic', ''),item.get('star', '')item.get('content', '')])
        elif data_type == 'news2016zh_train':
            merged_text = ' '.join([item.get('title', ''), item.get('content', ''),item.get('source', ''),item.get('time', '')])
        elif data_type == 'Bbaike_qa_train':
            merged_text = ' '.join([item.get('category', ''), item.get('title', ''), item.get('desc', ''),item.get('answer', '')])

        output_text_no_space = output_text.replace(" ", "")
        merged_text_no_space = merged_text.replace(" ", "")
        merged_text_clean = remove_html_tags_using_bs4(merged_text_no_space)

        if output_text_no_space in merged_text_clean:
            print(f'Partial match found in {data_type}: {item.get("qid", item.get("news_id", item.get("id", "")))}')
            occurence = occurence + 1
    
    return occurence

def remove_html_tags_using_bs4(html):
    soup = BeautifulSoup(html, "lxml")
    
    text_only = soup.get_text(separator=" ", strip=True)
    return text_only

def verify_samples(sample):
    # verification phase:
    # different types of datasets
    data_types = ['web_text_zh_train', 'Bbaike_qa_train', 'news2016zh_train']
    sum_occurence=0
    for data_type in data_types:
        sum_occurence = sum_occurence + check_partial_match(sample,data_type)
    return sum_occurence

def main():
    print(f"using device: {device}")

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    print("Loading GPT2...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name1)
    tokenizer.padding_side = 'left' 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.pad_token = tokenizer.eos_token

    model1 = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model1.resize_token_embeddings(len(tokenizer))
    model2 = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name2, return_dict=True).to(device)
    model1.eval()
    model2.eval()
    
    samples = []
    window_samples = []
    scores = {"XL": [], "S": [], "Sliding_window": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            #prompts = ["<|endoftext|>"] * args.batch-size
            prompts = [tokenizer.decode([model1.config.eos_token_id])] * args.batch_size
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            # batch generation
            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:
                # perplexity of GPT2-XL and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                p2 = calculatePerplexity(text, model2, tokenizer)

                # perplexity on sliding window sample
                p_window,window_text = calculate_window_perplexity(text, model1, tokenizer,args.window_size)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                window_samples.append(window_text)
                scores["XL"].append(p1)
                scores["S"].append(p2)
                scores["Sliding_window"].append(p_window)
                scores["zlib"].append(zlib_entropy)

            pbar.update(args.batch_size)

    scores["XL"] = np.asarray([item.cpu().numpy() for item in scores["XL"]])
    scores["S"] = np.asarray([item.cpu().numpy() for item in scores["S"]])
    scores["Sliding_window"] = np.asarray(scores["Sliding_window"])
    scores["zlib"] = np.asarray(scores["zlib"])

     # a naive de-duplication strategy
    idxs = pd.Index(samples)
    idxs_mask = ~(idxs.duplicated())
    print(idxs_mask)
    generated_samples_clean = np.asarray(samples)[idxs_mask]
    generated_samples_clean = generated_samples_clean.tolist()

    window_idxs = pd.Index(window_samples)
    window_idxs_mask = ~(window_idxs.duplicated())
    print(window_idxs_mask)
    # Filter out duplicates while preserving the list structure
    window_generated_samples_clean = [sample for i, sample in enumerate(window_samples) if window_idxs_mask[i]]

    scores["XL"] = scores["XL"][idxs_mask]
    scores["S"] = scores["S"][idxs_mask]
    scores["zlib"] = scores["zlib"][idxs_mask]
    scores["Sliding_window"] = scores["Sliding_window"][window_idxs_mask]

    assert len(generated_samples_clean) == len(scores["XL"])
    assert len(scores["S"]) == len(scores["XL"])
    assert len(window_generated_samples_clean) == len(scores["Sliding_window"])
    print("Num duplicates:", len(samples) - len(generated_samples_clean) + len(window_samples)- len(window_generated_samples_clean) )

    # Sort by perplexity
    metric = -np.log(scores["XL"])
    print(f"======== top sample by XL perplexity: ========")
    print_best(metric, generated_samples_clean, "PPL", scores["XL"])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores["S"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of S and XL perplexities: ========")
    print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "PPL-S", scores["S"])
    print()
    print()

    # Sort by sliding window perplexities 
    metric = -np.log(scores["Sliding_window"])
    print(f"======== top sample by sliding window perplexities: ========")
    print_best(metric, window_generated_samples_clean, "PPL-XL", scores["XL"], "PPL-XL-Sliding-window", scores["Sliding_window"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["XL"])
    print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "Zlib", scores["zlib"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--pretrained_model_name1', type=str, default=None, help="the target model name of the model from huggingface")
    parser.add_argument('--pretrained_model_name2', type=str, default=None, help="the name of the model for relative perplexity from huggingface")
    parser.add_argument('--window_size', type=int, default=50, help="the size of sliding window")
    parser.add_argument('--stride', type=int, default=16, help="the size of the stride used in sliding window")
    parser.add_argument('--dataset1', type=str, default=None, help="the dataset used for verification")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
