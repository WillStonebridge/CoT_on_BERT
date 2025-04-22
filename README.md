# Reasoning with Chain-of-Thought for Classification Models

## Instructions:
1. This homework evaluates your understanding of using Chain-of-Thought (CoT) generation and downstream classification.
2. Implement the training logic for both a T5-style generation model and a BERT-style classifier, completing the TODOs.
3. There are two files: `train.csv` and `test.csv`, located in the `data/` folder.
   - BERT will be trained on the `gold_label` from `train.csv`.
   - T5 (Flan-T5) will be trained on the `gold_CoT` from `train.csv`.
   - Load test data from `test.csv`.
   - Evaluate BERT by **saving** the answer as `pred_label` column in test data (label **should** only be A, B, C, or D).
   - **Add** column `bert_input` to test data which contains Question + Options (if no CoT) or Question + Options + CoT (if CoT).
   - **Save** the predictions in `data/test_pred_case.csv`.
   - Most of these steps are already implemented in `main.py`. You need to fill in the missing parts.
4. Description of data columns:
    - `question`: MMLU question
    - `choices`: option choices (total 4)
    - `subject`: MMLU topic
    - `gold_label`: correct option choice (for training BERT).
    - `gold_CoT`: The Chain-of-Thought reasoning (for training T5).
    - `pred_label`: The predicted label (for test).
    - `bert_input`: The input for BERT, which can be either Question + Options or Question + Options + CoT (for test).
5. Please use the packages from `requirements.txt`, with Python>=3.10. If you `import` any other libraries, please make sure to provide the list in `requirements.txt`.

Usage Examples:
```bash
# No training, no CoT
python main.py

# No training, with CoT
python main.py --cot

# Train only BERT with CoT
python main.py --bert_train --cot

# Train and predict using only BERT (no CoT)
python main.py --bert_train

# Train only T5 with CoT
python main.py --t5_train --cot

# Train both BERT and T5 with CoT
python main.py --bert_train --cot --t5_train
```

### Possible Behavior Table
| Case | Description                        | T5 Loaded | T5 Trained | CoT Used | BERT Trained |
|------|------------------------------------|-----------|------------|----------|---------------|
| 1    | No training, no CoT                | ❌        | ❌         | ❌       | ❌            |
| 2    | No training, with CoT              | ✅        | ❌         | ✅       | ❌            |
| 3    | Train only BERT with CoT           | ✅        | ❌         | ✅       | ✅            |
| 4    | Train only BERT without CoT        | ❌        | ❌         | ❌       | ✅            |
| 5    | Train only T5 with CoT             | ✅        | ✅         | ✅       | ❌            |
| 6    | Train both BERT and T5 with CoT    | ✅        | ✅         | ✅       | ✅            |

## Submission:
- Your submission should include:
  - `main.py`: with the training implementation for both T5 and BERT.
  - `data/`: with all the 6 `test_pred_case.csv` files.