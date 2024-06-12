txt_name = 'ja'

def calculate_total_lines(file_path):
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in file:
            total_lines += 1
    return total_lines

def split_file(file_path, num_splits, txt_name):
    total_lines = calculate_total_lines(file_path)
    lines_per_file = total_lines // num_splits
    extra_lines = total_lines % num_splits

    with open(file_path, 'r', encoding='utf-8') as file:
        for i in range(num_splits):
            part_lines = lines_per_file + (1 if i < extra_lines else 0)
            with open(f'{txt_name}_part_{i+1}.txt', 'w', encoding='utf-8') as part_file:
                for _ in range(part_lines):
                    line = file.readline()
                    # Remove specified characters from the line
                    line = line.replace("'", "").replace('"', "") \
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
                    part_file.write(line)

def calculate_total_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return len(file.readlines())


import pysubstringsearch

def create_suffix_array_for_each_part(num_splits):
    for i in range(num_splits):
        writer = pysubstringsearch.Writer(index_file_path=f'{txt_name}_part_{i+1}.idx')
        writer.add_entries_from_file_lines(f'{txt_name}_part_{i+1}.txt')
        writer.finalize()

file_path = f'{txt_name}.txt'
split_file(file_path,4,txt_name)
create_suffix_array_for_each_part(4)