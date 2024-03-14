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

def calcualte_window_perplexity(
    sentence: str,
    model,
    tokenizer,
    window_size: int = 50,
    stride: int = 16,
):
    ## input_ids == target_ids.
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    target_ids = input_ids.clone()

    ppls = []
    for idx in range(0, input_ids.size(1) - window_size, stride):
        ## Evaluate on a gpu.
        with torch.no_grad():
            outputs = model(
                input_ids[:, idx : idx + window_size].to(device),
                labels=target_ids[:, idx : idx + window_size].to(device),
            )

        ## And the perplexity is exponential of the loss of a sentence.
        loss, _ = outputs[:2]
        ppl = float(torch.exp(loss).cpu().detach().numpy())
        ppls.append(ppl)
    
    ## List "ppls" might be empty because of the full punctuations.
    return np.inf if ppls == [] else min(ppls)

def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        #for line in samples[idx].split("\n"):
        #    print(f"\t {line.rstrip()}")
        pprint(samples[idx])
        print()
        print()
        
def main():
    print(f"using device: {device}")

    # number of tokens to generate
    seq_len = 256

    # sample from the top_k tokens output by the model
    top_k = 40

    print("Loading GPT2...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained-model-name1)
    tokenizer.padding_side = 'left' 
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #tokenizer.pad_token = tokenizer.eos_token

    model1 = AutoModelForCausalLM.from_pretrained(args.pretrained-model-name1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model1.resize_token_embeddings(len(tokenizer))
    model2 = AutoModelForCausalLM.from_pretrained(args.pretrained-model-name2, return_dict=True).to(device)
    model1.eval()
    model2.eval()
    
    samples = []
    scores = {"XL": [], "S": [], "Sliding_window": [], "zlib": []}

    num_batches = int(np.ceil(args.N / args.batch-size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            # encode the prompts
            #prompts = ["<|endoftext|>"] * args.batch-size
            prompts = [tokenizer.decode([model1.config.eos_token_id])] * args.batch-size
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
                p_window = calcualte_window_perplexity(text, model1, tokenizer,args.window_size,args.stride)

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                samples.append(text)
                scores["XL"].append(p1)
                scores["S"].append(p2)
                scores["Sliding_window"].append(p_window)
                scores["zlib"].append(zlib_entropy)

            pbar.update(args.batch-size)

    scores["XL"] = np.asarray([item.cpu().numpy() for item in scores["XL"]])
    scores["S"] = np.asarray([item.cpu().numpy() for item in scores["S"]])
    scores["Sliding_window"] = np.asarray([item.cpu().numpy() for item in scores["Sliding_window"]])
    scores["zlib"] = np.asarray(scores["zlib"])

    # Sort by perplexity
    metric = -np.log(scores["XL"])
    print(f"======== top sample by XL perplexity: ========")
    print_best(metric, samples, "PPL", scores["XL"])
    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores["S"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of S and XL perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-S", scores["S"])
    print()
    print()

    # Sort by sliding window perplexities 
    metric = np.log(scores["Sliding_window"])
    print(f"======== top sample by sliding window perplexities: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "PPL-XL-Sliding-window", scores["Sliding_window"])
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["XL"])
    print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    print_best(metric, samples, "PPL-XL", scores["XL"], "Zlib", scores["zlib"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--pretrained-model-name1', type=str, default=None, help="the target model name of the model from huggingface")
    parser.add_argument('--pretrained-model-name2', type=str, default=None, help="the name of the model for relative perplexity from huggingface")
    parser.add_argument('--window_size', type=int, default=50, help="the size of sliding window")
    parser.add_argument('--stride', type=int, default=16, help="the size of the stride used in sliding window")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
