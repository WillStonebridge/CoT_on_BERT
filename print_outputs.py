import pandas as pd

if __name__ == "__main__":
    df_bert = pd.read_csv("data/test_pred_Bert_True_CoT_False_T5_False.csv")
    df_bert_cot = pd.read_csv("data/test_pred_Bert_True_CoT_True_T5_True.csv")

    for i in range(len(df_bert)):
        truth = df_bert["gold_label"][i]
        question = df_bert["question"][i]
        options = df_bert["choices"][i]
        non_cot_answer = df_bert["pred_label"][i]
        cot_answer = df_bert_cot["pred_label"][i]
        if truth == df_bert["pred_label"][i] and truth != df_bert_cot["pred_label"][i]:
            CoT = df_bert_cot["pred_CoT"][i]
            print(f"NON-COT CORRECT\nQuestion: {question}\nOptions: {options}\nCoT: {CoT}\nAnswers - bert: {non_cot_answer} CoT: {cot_answer} Truth: {truth}\n")
            breakpoint()
        elif truth != df_bert["pred_label"][i] and truth == df_bert_cot["pred_label"][i]:
            CoT = df_bert_cot["pred_CoT"][i]
            print(f"COT CORRECT\nQuestion: {question}\nOptions: {options}\nCoT: {CoT}\nAnswers - bert: {non_cot_answer} CoT: {cot_answer} Truth: {truth}\n")
            breakpoint()