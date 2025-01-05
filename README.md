# Predicting Missing Values in Survey Data Using Prompt Engineering

This repository contains the implementation of the method proposed in the paper:  
**"Predicting Missing Values in Survey Data Using Prompt Engineering for Addressing Item Non-Response"**  
Published in *Future Internet (2024)*.  
[Read the Paper](https://doi.org/10.3390/fi16100351)

---

## Overview

This project leverages **Large Language Models (LLMs)** to predict missing survey responses using prompt engineering techniques. The proposed method combines:
- **Row Selection**: Identifying similar respondents using cosine similarity.
- **Column Selection**: Selecting the most relevant question–answer pairs to enhance prediction context.

Compared to traditional imputation methods like MICE, MissForest, and TabTransformer, our approach:
- Achieves competitive or superior performance.
- Operates without complex preprocessing or additional training.
- Is scalable and adaptable to real-time survey analysis.

---

## Key Features
- **Row and Column Selection:** Efficiently selects few-shot examples and relevant context.
- **Prompt Engineering:** Generates tailored prompts for LLMs to predict item non-responses.
- **Lightweight Implementation:** Requires minimal preprocessing, enabling rapid inference.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/predicting-missing-values.git
    cd predicting-missing-values
    ```
2. Install dependencies:
    ```bash
    conda env create --file environment.yaml
    ```
## Code Overview

### 1. `langraph_naive.py`
- **Description**: Predicts missing survey responses using the `llama3-8B-Instruct` model.
- **Approach**: Selects survey questions with high vector similarity to the target question.
- **Method**: Implements the Naive Prompt method as described in the paper.

**Example Command**:
```bash
python langraph_naive.py --query_code SATIS_W116 --top_k 25 --n_shot 0 --output_file naive_results
```

### 2. `langraph_naive_gpt.py`
- **Description**: Predicts missing survey responses using `gpt-4-turbo` or `gpt-4o-mini` models.
- **Approach**: Similar to langraph_naive.py, this method selects questions based on vector similarity.
- **Method**: Implements the Naive Prompt method using GPT-based models.

**Example Command**:
```bash
python langraph_naive_gpt.py --query_code SATIS_W116 --top_k 25 --n_shot 0 --output_file naive_gpt_results
```

### 3. `langraph_cos_fixed_useful.py`
- **Description**: Predicts missing responses using `llama3-8B-Instruct`.
- **Approach**:
    - Generates a list of useful related questions to the target question.
    - Refines the context by selecting survey responses that match these useful questions.
- **Flexibility**:
    - Set n_shot=0 for **Non-Row Selection** setup.
    - Set n_shot ≥ 1 for **Full Context Method**.

**Example Command**:
```bash
python langraph_cos_fixed_useful.py --query_code SATIS_W116 --top_k 25 --n_shot 1 --output_file full_context_results
```

### 4. `langraph_cos_fixed_useful_gpt.py`
- **Description**: Similar to `langraph_cos_fixed_useful.py`, but uses `gpt-4-turbo` or `gpt-4o-mini`.
- **Approach**:
    - Generates and refines prompts with semantically related questions.
    - Combines advanced linguistic reasoning capabilities with relevant survey responses.
- **Flexibility**:
    - Set n_shot=0 for **Non-Row Selection** setup.
    - Set n_shot ≥ 1 for **Full Context Method**.

**Example Command**:
```bash
python langraph_cos_fixed_useful_gpt.py --query_code SATIS_W116 --top_k 25 --n_shot 2 --output_file useful_gpt_results
```

## Usage Details

### Command-Line Arguments
- `--query_code`: The code for the survey question being predicted (e.g., SATIS_W116).
- `--top_k`: Number of top-ranked related questions to include in the prompt.
- `--n_shot`: Number of few-shot examples (respondents) to include in the context.
- `--output_file`: Name of the file where results will be saved.

## Notes
1. Adjust the `--top_k` and `--n_shot` values to test different hyperparameter settings.
2. Results include F1-score evaluations and prompts used for predictions, saved in the specified `output_file`.
