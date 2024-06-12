import os

def resize_files_to_total_size(file_paths, ratios, total_size_mb):
    total_size_bytes = total_size_mb * 1024 * 1024  # Convert MB to bytes
    file_sizes = [os.path.getsize(f) for f in file_paths]
    total_size_current = sum(file_sizes)
    
    # Calculate how much to sample from each file based on the desired total size and current total size
    sample_sizes = [int((ratio / sum(ratios)) * total_size_bytes) for ratio in ratios]
    
    with open('combined_sample.txt', 'wb') as outfile:
        for file_path, sample_size in zip(file_paths, sample_sizes):
            with open(file_path, 'rb') as infile:
                read_size = 0
                while read_size < sample_size:
                    # Calculate how much to read without exceeding the sample size or the total size limit
                    chunk_size = min(4096, sample_size - read_size, total_size_bytes - outfile.tell())
                    if chunk_size <= 0:
                        break  # Stop if the total size limit is reached
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
                    read_size += len(chunk)

# Example usage
file_paths = ['web_text_zh_train_v3.txt', 'news2016zh_train_v3.txt', 'baike_qa_train_v3.txt']
ratios = [3, 7.5, 1.5]  # The size ratios between the files
total_size_mb = 350  # The total size of the output file in MB

resize_files_to_total_size(file_paths, ratios, total_size_mb)
