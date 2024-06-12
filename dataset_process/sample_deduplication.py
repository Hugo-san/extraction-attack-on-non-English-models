def remove_duplicates(input_file, output_file):
    seen = set()
    duplicate_count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            clean_line = line.strip()
            if clean_line in seen and clean_line != "":
                duplicate_count += 1
            else:
                seen.add(clean_line)
                outfile.write(clean_line + '\n')

    print(f"Number of duplicate lines: {duplicate_count}")

remove_duplicates('Sampled_Chinese_CC.txt', 'output.txt')
