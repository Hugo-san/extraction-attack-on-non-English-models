import pandas as pd
import pysubstringsearch

# load csv
csv_file_path = '128_zlib_best_samples_scores_German.csv'
data = pd.read_csv(csv_file_path)
index_file_path='sum_index_de.idx'
def check_partial_match(output_text,mask=0):
    occurrence = 0
    #output_text_no_space = output_text.replace(" ", "")
    reader = pysubstringsearch.Reader(index_file_path='sum_index_de.idx',)
    results = reader.search(output_text[mask:])
    occurrence += len(results)

    return occurrence

precision_count = 0

for index, row in data.iterrows():
    sample_text = row['Sample']

    # check the index file
    occ = 0
    occ += check_partial_match(sample_text,mask=15)
    
    data.at[index, 'Occurence'] = occ
    if occ>0:  
        precision_count += 1

data.to_csv(f'de_updated_{csv_file_path}', index=False)

precision = precision_count/100
print(precision)
