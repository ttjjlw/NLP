# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date
def get_f1score(result_file, target_file):
    '''
    result_file：get_result_file生成的文件
    target_file:格式同 result_file
    '''
    result = open(result_file, 'r', encoding='utf-8')
    target = open(target_file, 'r', encoding='utf-8')
    r_lines = result.readlines()
    t_lines = target.readlines()
    total_tags = 0  #target样本的字段数
    correct_tags = 0  #result中抽取出的正确字段数
    total_tab_tags = 0  #result中抽取出的字段数
    for r_line, t_line in zip(r_lines, t_lines):
        r_lis = r_line.split('  ') #每段标记两个空格隔开 如：北京/a  是个美丽的城市，鲁迅/b  曾经去过。
        t_lis = t_line.split('  ')
        for r_tag, t_tag in zip(r_lis, t_lis):
            if t_tag[-1] in ['a', 'b', 'c']:
                total_tags += 1
            if r_tag[-1] in ['a', 'b', 'c']:
                total_tab_tags += 1
                if r_tag[-1] == t_tag[-1] and len(r_tag) == len(t_tag):
                    correct_tags += 1
    recall = round(correct_tags / total_tags, 4)
    precise = round(correct_tags / total_tab_tags, 4)
    f1score = round(2 * recall * precise / (recall + precise), 4)
    result.close()
    target.close()
    return f1score
def compare_dg_result_file(output_dir=r'C:\Users\yxc\Desktop\Daguang'):
	'''
	output_dir:多个dg_result_file 存放的位置
	'''
	import os
	res_lis=os.listdir(output_dir)
	lis=[]
	for res01 in res_lis:
		result_file01=os.path.join(output_dir,res01)
		for res02 in res_lis:
			result_file02=os.path.join(output_dir,res02)
			if result_file01!=result_file02:
				f1score=get_f1score(result_file01,result_file02)
				print(f1score)
				if f1score<0.88:
					continue
				else:
					lis.append([res01,res02,f1score])
	print(lis)
compare_dg_result_file()
exit()

#隔两行变成隔一行
def twoline_to_1line_separation(line2_file,line1_file):
    '''
    :param line2_file: 一行一行的文本文件，有些文本之间隔了两行换行符
    :param line1_file: 一行一行的文本文件，把隔两行的换行符变成了隔一行
    :return:
    '''
    df=open(line1_file,'w',encoding='utf-8')
    with open (line2_file,'r',encoding='utf-8') as f:
        lines=f.readlines()
        k=0
        for line in lines:
            if line=='\n':
                k+=1
                if k==2:
                    df.write('\n')
                    k=0
                    continue
                else:
                    continue
            df.write(line)
    df.close()

# import pickle
# with open('data_path//DaGuang//dr_d_td.pkl', 'rb') as f:
#     id2word = pickle.load(f)
#     word2id = pickle.load(f)
#     _ = pickle.load(f)
# lis=[]
# with open (r'data_path\DaGuang\dg_test.txt','r',encoding='utf-8') as f:
#     lines=f.readlines()
#     for line in lines:
#         if line=='\n':
#             continue
#         word=line.split()[0]
#         if word not in word2id.keys():
#             lis.append(word)
#     s=set(lis)
# print(len(s))
#求列表中出现最多次数的那个元素
def most_list(lt):
    '''
    :param lt: 列表
    :return: 返回列表中出现最多次数的那个元素
    '''
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            most_str = i
            temp = lt.count(i)
    return most_str
def get_dg_train(train_dir,dg_train_dir):
    import codecs
    with codecs.open(train_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        results = []
        for line in lines:
            features = []
            tags = []
            samples = line.strip().split('  ')
            for sample in samples:
                sample_list = sample[:-2].split('_')
                tag = sample[-1]
                features.extend(sample_list)
                tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(
                    ['B-' + tag] + ['I-' + tag] * (len(sample_list) - 1))
            results.append(dict({'features': features, 'tags': tags}))
        # [{'features': ['7212', '17592', '21182', '8487', '8217', '14790', '19215', '4216', '17186', '6036',
        # '18097', '8197', '11743', '18102', '5797', '6102', '15111', '2819', '10925', '15274'],
        # 'tags': ['B-c', 'I-c', 'I-c', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        # 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}]
        train_write_list = []
        with codecs.open(dg_train_dir, 'w', encoding='utf-8') as f_out:
            for result in results:
                for i in range(len(result['tags'])):
                    train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
                train_write_list.append('\n')
            f_out.writelines(train_write_list)
import os
output_dir=r'C:\Users\yxc\Desktop\Daguang'
# output_dir1=r'C:\Users\yxc\Desktop\Daguang1'
res_lis=os.listdir(output_dir)
# for n in res_lis:
#     train_f=os.path.join(output_dir,n)
#     dg_train_f=os.path.join(output_dir,n)
#     get_dg_train(train_f,dg_train_f)
# exit()
lis=[]
for i in range(len(res_lis)):
    result_file=os.path.join(output_dir,res_lis[i])
    with open (result_file,'r') as f:
        lines01=f.readlines()
        lis.append(lines01)
result=[]
for line01,line02,line03,line04,line05 in zip(lis[0],lis[1],lis[2],lis[3],lis[4]):
    line=[]

    for tag1,tag2,tag3,tag4,tag5 in zip(line01.split(),line02.split(),line03.split(),line04.split(),line05.split()):
        ll=[]
        ll.append(tag1)
        ll.append(tag2)
        ll.append(tag3)
        ll.append(tag4)
        ll.append(tag5)
        line.append(most_list(ll))
    result.append('\t'.join(line))
with open ('dg_result_combine1.txt','w') as f:
    for line in result:
        f.write(line+'\n')

# import pickle
#
# with open('data_path//DaGuang//dr_d_td_all.pkl', 'rb') as f:
#     id2word = pickle.load(f)
#     word2id = pickle.load(f)
#     print('word2id的length:', len(word2id))
#     _ = pickle.load(f)
# with open('data_path//DaGuang//dg_vocal.txt',mode='a') as f:
#     for word in word2id.keys():
#         f.write(word+'\n')

# import matplotlib.pyplot as plt
# with open (r'D:\localE\code\daguang_extract\tensorflow\Chinese_ner_tensorflow-master\model_path\DaGuang\1566434267\results\result_file1','r') as f:
#     lines=f.readlines()
#     a=len(lines)
#     dic={}
#     count=0
#     for line in lines:
#         length=len(line.split('_'))
#         if length>100:
#             count+=1
#         if length not in dic:
#             dic[length]=1
#         else:
#             dic[length]+=1
# print(max(dic.keys()))
# print(count/a)
#
# plt.bar(dic.keys(),dic.values())
# plt.show()

# with open('data_path/DaGuang/corpus_all.txt','r') as f:
#     lines=f.readlines()
#     print(len(lines))
#     lis=[]
#     for line in lines:
#         if line=='\n':
#             print(1)
#         ll=line.split("_")
#         s=' '.join(ll)
#         lis.append(s)
#     print(len(lis))
# with open ('data_path/DaGuang/corpus_all_bert.txt','w') as f:
#     f.writelines(lis)