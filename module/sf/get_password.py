import pandas as pd
import random

def read_words_from_excel(file_path):
    df = pd.read_excel(file_path, header=None)
    words = df[0].dropna().tolist()
    words = [str(w).lower() for w in words]
    return words

def group_words_by_initial(words):
    word_groups = {}
    for word in words:
        try:
            initial = word[0]
        except:
            print(word)
        if initial not in word_groups:
            word_groups[initial] = []
        word_groups[initial].append(word)
    return word_groups

def generate_unique_keys(word_groups, num_keys):
    keys = set()
    while len(keys) < num_keys:
        selected_words = []
        initials_used = set()
        for initial in random.sample(list(word_groups.keys()), 12):
            word = random.choice(word_groups[initial])
            selected_words.append(word)
            initials_used.add(initial)
        key = ' '.join(selected_words)
        if key not in keys:
            keys.add(key)
    return list(keys)

def write_keys_to_txt(keys, file_path):
    with open(file_path, 'w') as f:
        for idx,key in enumerate(keys):
            f.write(str(idx+1)+'、' + key + '\n')

# 主程序
if __name__ == "__main__":
    seq = input('请输入保存文件的序列号，然后回车，例如：1：')
    num_keys = input('请输入要生成的密钥数量,然后回车：')
    file_path = '/Users/haoshuiyu/Documents/bs32.xlsx'  # Excel文件路径
    output_file_path = '/Users/haoshuiyu/Documents/keys_%s.txt'%seq.strip()  # 输出txt文件路径
    num_keys = int(num_keys.strip())  # 生成秘钥的数量

    words = read_words_from_excel(file_path)
    word_groups = group_words_by_initial(words)
    keys = generate_unique_keys(word_groups, num_keys)
    write_keys_to_txt(keys, output_file_path)

    print(f"{num_keys} unique keys have been generated and saved to {output_file_path}.")
