### BERT Config & Tokenizer (HF v4.56.0)

- BertConfig stores **all** hyper-params of BertModel (e.g. `hidden_size`, `initializer_range`).  
- BertTokenizer uses **WordPiece**; `BertTokenizerFast` offers Rust backend.  
- Speed test (800 tokens): **0.78×** vs normal tokenizer → Rust backend **18 % faster** on CPU.  