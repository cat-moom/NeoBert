from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import argparse
import os

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
        
def main(input_mode,model_path, input_file, save_dir, Peptide, HLA):
    if input_mode == 0:
        print('---------------------Arguments------------------------')
        print('model_path :',model_path)
        print('input_file :',input_file)
        print('save_dir :',save_dir)
        ensure_directory_exists(save_dir)
        print('------------------------------------------------------')
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_format = 'unknow'
        if str(input_file).endswith('.csv'):
            df = pd.read_csv(input_file, index_col=False)
            input_format = 'csv'
        else:
            df = pd.read_excel(input_file)
            input_format = 'excel'
        HLA_ls = df['MHC_sequence'].to_list()
        Peptide_ls = df['Peptide_sequence'].to_list()
        binding_prob_ls = []
        unbinding_ls = []
        binding_ls = []
        for index, row in df.iterrows():
            hla = row['MHC_sequence']
            pep = row['Peptide_sequence']
            text = hla + "[SEP]" + pep
            inputs = tokenizer(
                text,
                padding="max_length",
                max_length=128,
                return_token_type_ids=True,
                truncation=True,
                return_tensors="pt"
            )

            output = model(**inputs)
            logits = output.logits.squeeze()
            logits = torch.softmax(logits, dim=-1)
            binding_predict = logits.argmax(dim=-1).tolist()
            un_binding_predict = 1 - binding_predict
            prob = logits[binding_predict].cpu().item()
            un_binding_predict_prob = logits[un_binding_predict].cpu().item()
            print(logits)
            print(binding_predict)
            if binding_predict == 1:
                binding_prob_ls.append(prob)
                unbinding_ls.append(un_binding_predict_prob)
            else:
                binding_prob_ls.append(un_binding_predict_prob)#
                unbinding_ls.append(prob)
            binding_ls.append(binding_predict)
                

        data = {'MHC_sequence': HLA_ls, 'Peptide_sequence': Peptide_ls, 'un_binding_prob':unbinding_ls,'binding_prob': binding_prob_ls, 'predict': binding_ls}
        df_result = pd.DataFrame(data)
        
        if input_format == 'csv':
            predict_result_savepath = os.path.join(save_dir,'predict_output.csv')
            df_result.to_csv(predict_result_savepath, index=False)
        else:
            predict_result_savepath = os.path.join(save_dir,'predict_output.xlsx')
            df_result.to_excel(predict_result_savepath, index=False)
        print('Prediction results file saved in: ',predict_result_savepath)
        
    if input_mode == 1:
        print('---------------------Arguments------------------------')
        print('peptide sequence :',Peptide)
        print('HLA sequence :',HLA)
        print('------------------------------------------------------')
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        binding_prob_ls = []
        unbinding_ls = []
        binding_ls = []
        
        hla = HLA
        pep = Peptide
        text = hla + "[SEP]" + pep
        inputs = tokenizer(
            text,
            padding="max_length",
            max_length=128,
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
        )

        output = model(**inputs)
        logits = output.logits.squeeze()
        logits = torch.softmax(logits, dim=-1)
        binding_predict = logits.argmax(dim=-1).tolist()
        un_binding_predict = 1 - binding_predict
        prob = logits[binding_predict].cpu().item()
        un_binding_predict_prob = logits[un_binding_predict].cpu().item()
        print(logits)
        print(binding_predict)
        if binding_predict == 1:
            binding_prob_ls.append(prob)
            unbinding_ls.append(un_binding_predict_prob)
        else:
            binding_prob_ls.append(un_binding_predict_prob)#
            unbinding_ls.append(prob)
        binding_ls.append(binding_predict)
                

        data = {'MHC_sequence': HLA, 'Peptide_sequence': Peptide, 'un_binding_prob':unbinding_ls,'binding_prob': binding_prob_ls, 'predict': binding_ls}
        print('----------------预测结束------------------------')
        for key, value in data.items():
            print(f"Key: {key}")
            print(f"Value: {value}\n")
        
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for HLA-Peptide binding prediction")
    parser.add_argument('--input_mode', type=int, required=True,help='input mode----0:csv/xlsx/xls File--------1:Specific sequences')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    
    # input_mode:0
    parser.add_argument('--input_file', type=str, required=False, help='Path to the input CSV file')
    parser.add_argument('--save_dir', type=str, required=False, help='Path to save the output CSV file')
    
    # input_mode:1
    parser.add_argument('--Peptide', type=str, required=False, help='one peptide sequence')
    parser.add_argument('--HLA', type=str, required=False, help='one HLA sequence')

    args = parser.parse_args()
    # main(args.model_path, args.input_file, args.save_dir)
    main(args.input_mode,args.model_path, args.input_file, args.save_dir, args.Peptide, args.HLA)
