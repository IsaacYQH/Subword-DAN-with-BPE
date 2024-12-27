# Subword-DAN-with-BPE
Subword Deep Averaging Network with Byte Pair Encoding

## training scripts
all parameters used here are well documented in `.py` files.
```
# subword DAN model
python main.py\
    --model subwordDAN\
    --emb_size 30\
    --hidden_ratio 1.0\
    --epoch 300\
    --vocab_size 1000\
    --lr 1e-4