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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    for idx in range(0, input_ids.size(1) - window_size,16):
        window_ids = input_ids[:, idx : idx + window_size]
        
        # 转换为字符串检查是否有大量重复字符
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
        ppls.append(ppl)

    return np.inf if not ppls else min(ppls)

def has_repeated_chars(window_text, threshold=0.4):
    """检查文本窗口中是否有任何字符超过了指定的重复次数阈值"""
    from collections import Counter
    counts = Counter(window_text)
    most_common_cnt = counts.most_common(1)[0][1]
    if most_common_cnt / len(window_text) > threshold:
        return True
    return False

def print_best(metric, samples, name1, scores1, name2, scores2, n, file):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            file.write(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}\n")
        else:
            file.write(f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}\n")

        file.write("\n")
        pprint(samples[idx], stream=file)
        file.write("\n\n")

        
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
                p_window = calculate_window_perplexity(text, model1, tokenizer,args.window_size)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
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

    scores["XL"] = scores["XL"][idxs_mask]
    scores["S"] = scores["S"][idxs_mask]
    scores["zlib"] = scores["zlib"][idxs_mask]
    scores["Sliding_window"] = scores["Sliding_window"][idxs_mask]

    assert len(generated_samples_clean) == len(scores["XL"])
    assert len(scores["S"]) == len(scores["XL"])
    print("Num duplicates:", len(samples) - len(generated_samples_clean))

    with open('output.txt', 'w') as file:
        # Sort by perplexity
        metric = -np.log(scores["XL"])
        print(f"======== top sample by XL perplexity: ========\n", file=file)
        print_best(metric, generated_samples_clean, "PPL", scores["XL"], "PPL-S", scores["S"], 20, file)

        # Sort by ratio of log perplexities of S and XL models
        metric = np.log(scores["S"]) / np.log(scores["XL"])
        print(f"======== top sample by ratio of S and XL perplexities: ========\n", file=file)
        print_best(metric, generated_samples_clean, "PPL-S", scores["S"], "PPL-XL", scores["XL"], 20, file)

        # Sort by sliding window perplexities
        metric = -np.log(scores["Sliding_window"])
        print(f"======== top sample by sliding window perplexities: ========\n", file=file)
        print_best(metric, generated_samples_clean, "PPL-XL-Sliding-window", scores["Sliding_window"], None, None, 20, file)

        # Sort by ratio of Zlib entropy and XL perplexity
        metric = scores["zlib"] / np.log(scores["XL"])
        print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========\n", file=file)
        print_best(metric, generated_samples_clean, "Zlib", scores["zlib"], "PPL-XL", scores["XL"], 20, file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--pretrained_model_name1', type=str, default=None, help="the target model name of the model from huggingface")
    parser.add_argument('--pretrained_model_name2', type=str, default=None, help="the name of the model for relative perplexity from huggingface")
    parser.add_argument('--window_size', type=int, default=50, help="the size of sliding window")
    parser.add_argument('--stride', type=int, default=16, help="the size of the stride used in sliding window")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
