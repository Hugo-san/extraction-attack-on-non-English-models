import pandas as pd
import pysubstringsearch

# load csv
csv_file_path = '128_zlib_best_samples_scores_Ja.csv'
data = pd.read_csv(csv_file_path)

def check_partial_match(output_text,mask):
    occurence = 0
    
    output_text_no_space = output_text.replace("'", "").replace('"', "") \
                               .replace('“', '').replace('”', '') \
                               .replace('‘', '').replace('’', '') \
                               .replace('(', '').replace(')', '') \
                               .replace('（', '').replace('）', '') \
                               .replace(' ', '').replace(',', '') \
                               .replace('!', '').replace('！', '')\
                               .replace('?', '').replace('？', '')\
                               .replace('【', '').replace('[', '')\
                               .replace('】', '').replace(']', '')\
                               .replace('\n', '')
    
    reader = pysubstringsearch.Reader(index_file_path=f'ja_v2.idx',)
    results = reader.search(output_text_no_space[mask:])
    occurence += len(results)

    return occurence

precision_count = 0

for index, row in data.iterrows():
    sample_text = row['Sample']
    # check the index file
    occ = 0
    occ += check_partial_match(sample_text,mask=5)
    
    data.at[index, 'Occurence'] = occ
    if occ>0:  
        precision_count += 1

data.to_csv(f'updated_{csv_file_path}', index=False)

precision = precision_count/100
print(precision)
