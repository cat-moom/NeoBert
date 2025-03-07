import os
import pandas as pd

OrinDir = './data_Folders/'


# 定义一个函数，生成所有循环移动后的 Peptide（不包括初始 Peptide）
def generate_all_rotations(peptide):
    return [peptide[i:] + peptide[:i] for i in range(1, len(peptide))]

for i in os.listdir(OrinDir):
    if 'fake' not in i:
        file_path = OrinDir+i
        # print('文件:',i,'   路径:',file_path)
        df = pd.read_csv(file_path,index_col=False)

        # 创建新的 DataFrame，用于存储所有可能的循环移动数据
        new_rows = []
        # 遍历原始 DataFrame 的每一行，生成新的行
        for idx, row in df.iterrows():
            rotations = generate_all_rotations(row['Peptide'])
            for rotated_peptide in rotations:
                new_row = row.copy()
                new_row['Peptide'] = rotated_peptide
                new_row['binding'] = 0  # 将 binding 列的值设置为 0
                new_rows.append(new_row)

        # 将新的行创建成一个新的 DataFrame
        new_df = pd.DataFrame(new_rows)
        new_df.to_csv(OrinDir+'fake_'+i,index=False)
        # print(new_df)
        print(i,'done')
