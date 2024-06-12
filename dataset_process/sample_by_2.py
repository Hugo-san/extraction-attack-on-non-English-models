def delete_half_of_file(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    
    keep_lines_count = len(lines) // 2  

    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines[:keep_lines_count])

delete_half_of_file('your_file.txt')
