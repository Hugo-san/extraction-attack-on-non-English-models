data_types = ['web_text_zh_train', 'baike_qa_train', 'news2016zh_train']

for data_type in data_types:
    input_filename = f'{data_type}_no_space.txt'
    output_filename = f'{data_type}_no_space_lowercase.txt'
    with open(input_filename, 'r', encoding='utf-8') as infile, \
        open(output_filename, 'w', encoding='utf-8') as outfile:
        for line in infile:
            processed_line = line.replace(" ", "")
            processed_line = processed_line.lower()
            outfile.write(processed_line)

import pysubstringsearch

# creating a new index file
# if a file with this name is already exists, it will be overwritten
writer = pysubstringsearch.Writer(
    index_file_path='sum_index.idx',
)

# adding entries from file lines
for data_type in data_types:
    writer.add_entries_from_file_lines(f'{data_type}_no_space_lowercase.txt')

# making sure the data is dumped to the file
writer.finalize()