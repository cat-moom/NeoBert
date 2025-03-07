import json
import os
import pandas as pd

dir = './process_data'
ls_all_test_data = []
for i in os.listdir(dir):
    if i.endswith('.csv'):
        file_path = dir + '\\' + i
        df = pd.read_csv(file_path)
        file_save_name = dir + '\\' + i.split('.csv')[0] + '.json'
        ls_save = []
        for index, row in df.iterrows():
            MHC_sequence = row['HLA_generate']
            Peptide = row['Peptide']
            text = MHC_sequence + '[SEP]' + Peptide
            label = row['binding']
            ls_save.append({'text': text, 'label': label})

        # 将列表数据写入JSON文件
        with open(file_save_name, 'w', encoding='utf-8') as file:
            json.dump(ls_save, file, ensure_ascii=False, indent=4)

        print(f"数据已成功保存到文件: {file_save_name}")
#         ls_all_test_data = ls_all_test_data + ls_save
# with open('E:\\517课题备份\\数据集处理_519\\train_dataset\\dataset_json\\test_all.json', 'w', encoding='utf-8') as file:
#     json.dump(ls_all_test_data, file, ensure_ascii=False, indent=4)
# print('done!')


