import pysubstringsearch

txt_name = 'ja'

def create_index_for_large_file(file_path, txt_name):
    writer = pysubstringsearch.Writer(index_file_path=f'{txt_name}_v2.idx')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        passage = []
        
        for line in file:
            # If it's an empty line, consider it as the end of a passage
            if not line.strip():
                # Process the accumulated passage
                if passage:
                    passage_text = ''.join(passage)
                    passage_clean = passage_text.replace("'", "").replace('"', "") \
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
                    
                    if passage_clean:
                        writer.add_entry(passage_clean)
                    
                    # Clear the passage after processing
                    passage = []
            
            else:
                # Add the line to the current passage
                passage.append(line)
        
        # Handle the last passage if it wasn't followed by an empty line
        if passage:
            passage_text = ''.join(passage)
            passage_clean = passage_text.replace("'", "").replace('"', "") \
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
            
            if passage_clean:
                writer.add_entry(passage_clean)

    writer.finalize()

file_path = f'{txt_name}.txt'
create_index_for_large_file(file_path, txt_name)
