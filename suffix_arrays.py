import json
from bs4 import BeautifulSoup
def load_data(json_file, data_type):
  data = []
  with open(json_file, 'r', encoding='utf-8') as file:
    for line in file:
      data.append(json.loads(line.strip()))
    return data

def remove_html_tags_using_bs4(html):
    soup = BeautifulSoup(html, "lxml")
    
    text_only = soup.get_text(separator=" ", strip=True)
    return text_only

def merge_text_and_write_to_txt(json_data, data_type, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in json_data:
            merged_text = ''
            if data_type == 'web_text_zh_train':
                merged_text = ' '.join([item.get('title', ''), item.get('desc', ''), item.get('topic', ''),str(item.get('star', '')),item.get('content', '')]).strip()
            elif data_type == 'news2016zh_train':
                merged_text = ' '.join([item.get('title', ''), item.get('content', ''),item.get('source', ''),str(item.get('time', ''))]).strip()
            elif data_type == 'baike_qa_train':
                merged_text = ' '.join([item.get('category', ''), item.get('title', ''), item.get('desc', ''),item.get('answer', '')]).strip()
            
            merged_text_clean = remove_html_tags_using_bs4(merged_text)
            merged_text_no_space = merged_text_clean.replace(" ", "")
            merged_text_no_space = merged_text_no_space.lower()
            file.write(merged_text_no_space + '\n')

data_types = ['web_text_zh_train', 'baike_qa_train', 'news2016zh_train']

for data_type in data_types:
    json_file = f'{data_type}.json'
    json_data = load_data(json_file, data_type)
    output_file = f'{data_type}.txt'
    merge_text_and_write_to_txt(json_data, data_type, output_file)

import pysubstringsearch

# creating a new index file
# if a file with this name is already exists, it will be overwritten
writer = pysubstringsearch.Writer(
    index_file_path='output.idx',
)

# adding entries from file lines
for data_type in data_types:
    writer.add_entries_from_file_lines(f'{data_type}.txt')

# making sure the data is dumped to the file
writer.finalize()
