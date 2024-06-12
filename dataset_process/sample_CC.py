def truncate_file_to_size(file_path, target_size_mb):
    target_size_bytes = target_size_mb * 1024 * 1024  
    buffer = ''  
    read_size = 0  

    
    with open(file_path, 'rb') as file:
        while read_size < target_size_bytes:
            
            chunk = file.read(min(4096, target_size_bytes - read_size))
            if not chunk:
                break
            buffer += chunk.decode('utf-8', errors='ignore')  
            read_size += len(chunk)

    
    with open("Sampled_Ja_CC.txt", 'w', encoding='utf-8') as file:
        file.write(buffer)


truncate_file_to_size('ja.txt', 350)
