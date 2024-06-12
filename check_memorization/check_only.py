import pandas as pd
import pysubstringsearch

# load csv
csv_file_path = '256_zlib_best_samples_scores_German.csv'
data = pd.read_csv(csv_file_path)
index_file_path='sum_index_de.idx'
def check_partial_match(output_text):
    occurrence = 0
    
    reader = pysubstringsearch.Reader(index_file_path='sum_index_de.idx',)
    results = reader.search(output_text)
    occurrence += len(results)

    return occurrence

precision_count = 0

for index, row in data.iterrows():
    sample_text = row['Sample']
    parts = sample_text.split('\n', maxsplit=1)

    first_part = parts[0]
    second_part = parts[1] if len(parts) > 1 else ""
    # check the index file
    occ_1 = 0
    occ_1 += check_partial_match(first_part)
    occ_2 = 0
    if second_part != "":
        occ_2 += check_partial_match(second_part)
    
    data.at[index, 'occurence_1'] = occ_1
    if occ_1:  
        precision_count += 1

    data.at[index, 'occurence_2'] = occ_2
    if occ_2 != 0 and occ_1 == 0:
        precision_count += 1

data.to_csv(f'updated_{csv_file_path}', index=False)

precision = precision_count/100
print(precision)
