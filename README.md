````markdown
# CEPTA-based Transformer Language Model

A project to experiment with a CEPTA-based Transformer language model using DeepSeek-V3 talknizer.
Combining path-specific perceptron (CEPTA) and low-rank state space model (SSM)-based time series mixer, language modeling is performed with lightweight context modules instead of traditional attention.

---

## Key components

### 1. Talk Nizer

- Use 'deepseek-ai/deepseek-V3' talk nizer
- Securely initialize padding tokens and padding side settings
- Provide batch encoding and decoding utility

```python
from tokenizer import get_deepseek_v3_tokenizer

tokenizer = get_deepseek_v3_tokenizer()
encoded = tokenizer( 
["hello world"], 
return_tensors="pt", 
padding=True, 
truncation=True; 
max_length=128;
)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
````

### 2. CEPTA Perceptron

* path-wise perceptron with number of paths 'P' and abbreviated dimension 'alpha'
*Two Input Modes 

* Dense mode: '(B, T, d)' Enter the real feature 
* Index mode: '(B, T)' Token ID input (operates like an embedding table)
* Hard gate + Straight-Through Estimator (STE) option
* Includes local learning rule (Oja-style) and homeostasis (SP) utilities

Output:

* 'U': path potential, shape '(B, T, P)'
* 'F': Hard Gate (0/1), shape '(B, T, P)'
* `Y`: fan-out output, shape `(B, T, P, alpha)`

### 3. Low-Rank Cross-Path SSM (`CeptaSSMLowRank`)

* Model the time series interactions between paths as a low-rank state space

* Default update formula (summary): 

* `r_t = t_t @ V_r` 
* `a_t = sigmoid(r_t @ W_lambda + b_lambda)` 
* `s_t = a_t * s_{t-1} + (F_t ⊙ t_t) @ V_b / sqrt(P_r)` 
* `t_tilde = s_t @ V_o`

* Input: 't' (usually 'F ⊙ U') optional gate 'F', state cache

* Output: 't_tilde' mixed with time axis and last status cache

### 4. Token Embedding ('CepaToken Embedding')

Pipeline:

1. Token ID input ‘(B, T)’
2. CEPTA index pathway: `input_ids -> (U, F, Y)`
3. Calculate 't = F ⊙ U'
4. Time axis mix: `t_tilde` with `CeptaSSMLowRank`
5. Convert hidden vector to `Linear(P -> d_model)`
6. Addition of sinusoidal positional encoding

Output:

* Transformer input hidden '(B, T, d_model) '
* internal intermediate results dict: '{"U", "F", "t", "t_tilde", "Y"}'

### 5. CEPTA Transformer Block (`CeptaTransformerBlock`)

* Pre-LN structure: 

1. `RMSNorm` 
2. `CeptaContextBlock` (dense CEPTA + SSM) 
3.Residual add 
4. `RMSNorm` 
5. GELU MLP (`FeedForward`) 
6. Residual add

`CeptaTransformerLM`:

* `CeptaTokenEmbedding` + `n_layers` `CeptaTransformerBlock`
* Calculate vocablogits with 'lm_head' after the last 'RMSNorm'
* Input: 'input_ids (B, T)'
* Output: `logits (B, T, vocab_size)`

---

## File Structure

* `tokenizer.py` 
DeepSeek-V3 torque generator loader and encoding/decoding utility.

* `perceptron_cepta.py` 
CEPTA perceptron core implementation (dense/index path, custom autograd, local update).

* `cepta_ssm.py` 
Low-rank cross-path SSM layer ('CepaSSMLowRank').
* `embedding.py` 

* 'Cepa EmbeddingConfig': Embedding Settings dataclass 
* 'CepaTokenEmbedding': Token → CEPTA → SSM → hidden vector pipeline 
* Sinusoidal positional encoding

* `module_layer.py` 

* `RMSNorm`, `FeedForward` 
* `CeptaContextBlock`, `CeptaTransformerBlock` 
* 'CepaTransformerLM': full language model

* `main.py` 

* Entry Points for Demos 
* Create console-based interactive after model/talker build

* `train_test.py` 

* CEPTA Transformer LM Learning/Evaluation Script 
* Load Dataset, generate talkwise, block-wise LM Dataset 
* Save learning loops, evaluation loops, checkpoints

---

## Environment and Installation

### Requirements

*Python 3.x
* PyTorch
* Transformers (Hugging Face)

Example:

```bash
pip install torch transformers
```

Running in a CUDA environment significantly improves the learning speed.
For PyTorch installation options (such as CUDA version), you should refer to the official documentation to select the appropriate environment.

---

## Data preparation

By default, 'train_test.py' expects the following directories:

```text
Z:/Final_project/data_set/After_aling
```

Read all '.txt' files under that directory in sorted name order and attach them to a single token sequence.
Example:

```text
After_aling/ 
01.txt 
02.txt 
03.txt 
...
```

If you want to use a different path, you can change it through the '--data_dir' factor.

```bash
python train_test.py --data_dir path/to/your_dataset
```

---

## Run Learning

An example of running a basic learning script is as follows.

```bash
python train_test.py \ 
--data_dir path/to/your_dataset \ 
--epochs 3 \ 
--batch_size 2 \ 
--block_size 512 \ 
--lr 1e-4 \ 
--dtype_store bf16 \ 
--train_ratio 0.9 \ 
--save_path septa_lm.pt
```

Key factor description:

* '--data_dir': Directory with learning text (.txt) file
* '--epochs': number of epochs
* '--batch_size': batch size
* '--block_size': token length of one sample (LM context length)
* '--lr': Learning rate
* '--dtype_store': CEPTA parameter storage dtype ('bf16', 'fp16', 'fp32')
* '--train_ratio': the proportion of all tokens to be allocated to train
* '--save_path': the path to store model weights after completion of training

During learning, training loss and test loss are output for each epoch, and 'state_dict' is stored in the last specified path.

---

## Interactive Text Generation Demo

'main.py ' is a demo that allows you to enter prompts and check the model output text through a simple console interface.

```bash
python main.py
```

After execution:

```text
>>> (Enter prompt)
```

Generates tokens up to 'max_new_tokens' for the entered prompt,
The current code uses a random initialized model.

If you want to load and use the learned checkpoints, you can modify them as follows after the model build of 'main.py '.

```python
model, tokenizer, device, max_len = build_model()
state = torch.load("cepta_lm.pt", map_location=device)
model.load_state_dict(state)
model.eval()
```

---

## Model Settings

The default settings used by 'main.py ' and 'train_test.py' are as follows.

* 'P = 512' (number of paths)
* 'alpha = 4' (fan-out dimension)
* 'P_r = 64' (SSM status rank)
* `d_model = 1024` (Transformer hidden dim)
* `n_layers = 6` (number of Transformer blocks)
* 'max_seq_len = 512' (max sequence length)
* 'dtype_store = "bf16" (CEPTA parameter storage dtype, example)

These values ​​can be adjusted in the 'CepaEmbeddingConfig' and the learning script.

---

## Licenses

This project will be distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
For more information, see the file 'LICENSE' in the repository.

```
```
