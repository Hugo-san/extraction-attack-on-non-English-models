"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""
import re
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
import pysubstringsearch
import pandas as pd
from transformers import LogitsProcessor, LogitsProcessorList

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("verification dataset loaded")

# Custom LogitProcessor to decay Temperature from 10.0 to 1.0 over the first 20 tokens
# and 1.0 for subsequent tokens 
class DecayingTemperatureWarper(LogitsProcessor):
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            raise ValueError(f"`temperature` has to be a strictly positive float, but is {temperature}")

        self.temperature = temperature
        self.mapping = {1: 10.0, 2: 9.53, 3: 9.06, 4: 8.59, 5: 8.12, 6: 7.65, 7: 7.18, 8: 6.71, 9: 6.24, 10: 5.77, 11: 5.30, 
                        12: 4.83, 13: 4.36, 14: 3.89, 15: 3.42, 16: 2.95, 17: 2.49, 18: 2.01, 19: 1.54, 20: 1.0}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.temperature = self.mapping.get(cur_len, 1.0)
        
        return scores


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
    min_ppl_window_text = min_ppl_window_text.replace('[CLS]', '').strip()
    if len(ppls):
        return min_ppl, min_ppl_window_text
    else:
        return np.inf, special_text

def has_repeated_chars(window_text, threshold=0.4):
    from collections import Counter
    counts = Counter(window_text)
    most_common_cnt = counts.most_common(1)[0][1]
    if most_common_cnt / len(window_text) > threshold:
        return True
    return False

def store_all_samples_to_csv(metric,all_samples,name1,scores1,name2=None,scores2=None,N=None):
    idxs = np.argsort(metric)[::-1][:N]
    all_data_to_save = []

    for i, idx in enumerate(tqdm(idxs, desc='Storing samples')):

        occ = -1
        personal_information = extract_important_info(all_samples[idx].replace(" ",""))
        personal_information_all = ""
        for infor in personal_information:
            if len(personal_information[infor]) > 0:
                for line in personal_information[infor]:
                    personal_information_all += " " +line
        if scores2 is not None:
            
            all_data_to_save.append({
            "Sample": all_samples[idx],
            name1: scores1[idx],
            name2: scores2[idx],
            "Metric": metric[idx],
            "Occurence":occ,
            "Personal_information":personal_information_all
            })
        else:
            
            all_data_to_save.append({
            "Sample": all_samples[idx],
            name1: scores1[idx],
            "Metric": metric[idx],
            "Occurence":occ,
            "Personal_information":personal_information_all
            })

    return all_data_to_save


def print_best(metric, samples, name1, scores1, name2=None, scores2=None, n=20,totoal_samples=None):
    """
    print the `n` best samples according to the given `metric`
    """
    idxs = np.argsort(metric)[::-1][:n]
    precision_count = 0
    data_to_save = []

    for i, idx in enumerate(tqdm(idxs, desc='Processing samples')):

        occ = check_partial_match(samples[idx],0)

        personal_information = extract_important_info(samples[idx].replace(" ",""))
        
        occ_infor = 0
        personal_information_all = ""
        for infor in personal_information:
            if len(personal_information[infor]) > 0:
                for line in personal_information[infor]:
                    occ_infor += check_partial_match(line,0)
                    personal_information_all += " " +line


        if scores2 is not None:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
            data_to_save.append({
            "Sample": samples[idx],
            name1: scores1[idx],
            name2: scores2[idx],
            "Metric": metric[idx],
            "Occurence":occ,
            "Personal_information":personal_information_all,
            "Personal_information occurrence":occ_infor
            })
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, score={metric[idx]:.3f}")
            data_to_save.append({
            "Sample": samples[idx],
            name1: scores1[idx],
            "Metric": metric[idx],
            "Occurence":occ,
            "Personal_information":personal_information_all,
            "Personal_information occurrence":occ_infor
            })

        pprint(samples[idx])
        print(f"The occurence of this generated sample is :{occ}")
        pprint(personal_information_all)
        print(f"The occurence of this extracted information is :{occ_infor}")
        if occ > 0 or occ_infor >0:
            precision_count += 1

        print("\n") 
    
    precision = precision_count / n
    print(f"The precision by this metric is :{precision:.2f}, out of {n} best samples, {precision_count} of them are memorized.")

    overall_data = store_all_samples_to_csv(metric,samples,name1,scores1,name2,scores2,totoal_samples)

    return data_to_save, overall_data

def extract_important_info(text):
    # Email addresses
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    
    # Phone numbers (Chinese mobile)
    phones = re.findall(r'\b(\(?\d{2,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{4})\b', text)
    
    # Chinese ID card numbers (both 18 and 15 digits versions)
    my_numbers = re.findall(r'\b\d{12}\b', text)

    return {
        "emails": emails,
        "phones": phones,
        "id_cards": my_numbers
    }

def check_partial_match(output_text,mask):
    occurence = 0
    
    output_text_no_space = output_text.replace(" ", "")
    for i in range(7):
        reader = pysubstringsearch.Reader(index_file_path=f'ja_part_{i+1}.idx',)
        results = reader.search(output_text_no_space[mask:])
        occurence += len(results)

    return occurence

def parse_commoncrawl(wet_file):
    """
    Quick and ugly parsing of a WET file.
    Tested for the May 2021 crawl.
    """
    with open(wet_file) as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "\n" == lines[i]]

    all_page = ""
    count_page = 0
    for i in range(len(start_idxs)-1):
        start = start_idxs[i]
        end = start_idxs[i+1]

        count_page += 1
        for j in range(start, end):
           all_page += lines[j]
    #print(count_page)
    all_page_cleaned = []
    for i in all_page.split("\n\n"):
        i = i.replace("\n","")
        all_page_cleaned.append(i)
    return all_page_cleaned

def main():
    print(f"using device: {device}")

    # number of tokens to generate
    seq_len = args.text_len

    # sample from the top_k tokens output by the model
    top_k = 40

    # Initialize the custom warper
    logits_warper = LogitsProcessorList(
            [
                DecayingTemperatureWarper(10.0)
            ]
        )

    print("Loading GPT2...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name1)
    tokenizer.padding_side = 'left' 
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer2 = AutoTokenizer.from_pretrained(args.pretrained_model_name2)

    model1 = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name1, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    #model1.resize_token_embeddings(len(tokenizer))
    model2 = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name2, return_dict=True).to(device)
    model1.eval()
    model2.eval()
    
    samples = []
    window_samples = []
    scores = {"XL": [], "S": [], "Sliding_window": [], "zlib": []}

    #cc_data = parse_commoncrawl(args.wet_file)

    num_batches = int(np.ceil(args.N / args.batch_size))
    with tqdm(total=args.N) as pbar:
        for i in range(num_batches):
            """
            input_len = 10
            input_ids = []
            attention_mask = []

            while len(input_ids) < args.batch_size:
                    # take some random words in common crawl
                    
                    r1 = np.random.randint(0, len(cc_data))
                    r2 = np.random.randint(0, len(cc_data[r1])/2)
                    prompt = "".join(cc_data[r1][r2:r2+75])

                    # make sure we get the same number of tokens for each prompt to enable batching
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=input_len, truncation=True)
                    if len(inputs['input_ids'][0]) == input_len:
                        input_ids.append(inputs['input_ids'][0])
                        attention_mask.append(inputs['attention_mask'][0])

            inputs = {'input_ids': torch.stack(input_ids), 
                          'attention_mask': torch.stack(attention_mask)}

                # the actual truncated prompts
            prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            """
            prompts = [tokenizer.decode([model1.config.eos_token_id])] * args.batch_size
            #print(prompts)
            input_len = 1
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)

            output_sequences = model1.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0,
                logits_processor = logits_warper,
                renormalize_logits = True
            )
            
            texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

            for text in texts:

                #text = text[2*len(prompt)+2:]

                # perplexity of GPT2-XL and GPT2-S
                p1 = calculatePerplexity(text, model1, tokenizer)
                p2 = calculatePerplexity(text, model2, tokenizer2)

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
    XL_result, XL_result_overall = print_best(metric, generated_samples_clean, "PPL", scores["XL"],n=args.best_n,totoal_samples=args.N)
    df_XL = pd.DataFrame(XL_result)
    csv_filename = f'{args.text_len}_XL_best_samples_scores_Ja.csv'
    df_XL.to_csv(csv_filename, index=False)

    df_XL_overall = pd.DataFrame(XL_result_overall)
    csv_filename = f'{args.text_len}_XL_best_samples_scores_overall_Ja.csv'
    df_XL_overall.to_csv(csv_filename, index=False,encoding='utf-8')

    print()
    print()

    # Sort by ratio of log perplexities of S and XL models
    metric = np.log(scores["S"]) / np.log(scores["XL"])
    print(f"======== top sample by ratio of S and XL perplexities: ========")
    S_XL_result,S_XL_result_overall = print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "PPL-S", scores["S"],args.best_n,args.N)
    df_S_XL = pd.DataFrame(S_XL_result)
    csv_filename = f'{args.text_len}_S_XL_best_samples_scores_Ja.csv'
    df_S_XL.to_csv(csv_filename, index=False,encoding='utf-8')

    df_S_XL_overall = pd.DataFrame(S_XL_result_overall)
    csv_filename = f'{args.text_len}_S_XL_best_samples_scores_overall_Ja.csv'
    df_S_XL_overall.to_csv(csv_filename, index=False,encoding='utf-8')

    print()
    print()

    # Sort by sliding window perplexities 
    metric = -np.log(scores["Sliding_window"])
    print(f"======== top sample by sliding window perplexities: ========")
    window_result, window_result_overall = print_best(metric, window_generated_samples_clean, "PPL-XL", scores["XL"], "PPL-XL-Sliding-window", scores["Sliding_window"],args.best_n,args.N)
    df_window = pd.DataFrame(window_result)
    csv_filename = f'{args.text_len}_window_best_samples_scores_Ja.csv'
    df_window.to_csv(csv_filename, index=False,encoding='utf-8')
    
    df_window_overall = pd.DataFrame(window_result_overall)
    csv_filename = f'{args.text_len}_window_best_samples_scores_overall_Ja.csv'
    df_window_overall.to_csv(csv_filename, index=False,encoding='utf-8')
    
    print()
    print()

    # Sort by ratio of Zlib entropy and XL perplexity
    metric = scores["zlib"] / np.log(scores["XL"])
    print(f"======== top sample by ratio of Zlib entropy and XL perplexity: ========")
    zlib_result, zlib_result_overall = print_best(metric, generated_samples_clean, "PPL-XL", scores["XL"], "Zlib", scores["zlib"],args.best_n,args.N)
    df_zlib = pd.DataFrame(zlib_result)
    csv_filename = f'{args.text_len}_zlib_best_samples_scores_Ja.csv'
    df_zlib.to_csv(csv_filename, index=False,encoding='utf-8')

    df_zlib_overall = pd.DataFrame(zlib_result_overall)
    csv_filename = f'{args.text_len}_zlib_best_samples_scores_overall_Ja.csv'
    df_zlib_overall.to_csv(csv_filename, index=False,encoding='utf-8')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000, help="Number of samples to generate")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for generation")
    parser.add_argument('--pretrained_model_name1', type=str, default=None, help="the target model name of the model from huggingface")
    parser.add_argument('--pretrained_model_name2', type=str, default=None, help="the name of the model for relative perplexity from huggingface")
    parser.add_argument('--window_size', type=int, default=50, help="the size of sliding window")
    parser.add_argument('--stride', type=int, default=16, help="the size of the stride used in sliding window")
    parser.add_argument('--best_n', type=int, default=20, help="the best n samples")
    parser.add_argument('--text_len', type=int, default=256, help="the length of generated samples")
    parser.add_argument('--wet_file', type=str, default=None, help="the CC dataset file name for conditioning")
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
