# Chatterbox Turkish Fine-Tuning Guide

## 1. Setup (local machine)

```bash
# Clone fine-tuning repo next to chatterbox-vllm-streaming
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git \
    /path/to/chatterbox-finetuning

# Create dataset directory
mkdir -p chatterbox-finetuning/MyTTSDataset/wavs
```

## 2. Prepare data (local machine)

Convert FLEURS TSV to LJSpeech format:

```python
import csv

tsv_path = '/path/to/fleurs-turkish/train.tsv'
out_path = '/path/to/chatterbox-finetuning/MyTTSDataset/metadata.csv'

rows = []
with open(tsv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) < 7:
            continue
        basename = row[1].strip().replace('.wav', '')
        raw_text = row[2].strip()
        norm_text = row[3].strip()
        rows.append((basename, raw_text, norm_text))

with open(out_path, 'w', encoding='utf-8', newline='') as f:
    for basename, raw, norm in rows:
        f.write(f'{basename}|{raw}|{norm}\n')
```

Symlink wav files:

```bash
ln -s /path/to/fleurs-turkish/train/*.wav chatterbox-finetuning/MyTTSDataset/wavs/
```

## 3. Transfer to GPU machine

```bash
rsync -avP chatterbox-finetuning/MyTTSDataset/wavs/ user@gpu-host:chatterbox-finetuning/MyTTSDataset/wavs/
scp chatterbox-finetuning/MyTTSDataset/metadata.csv user@gpu-host:chatterbox-finetuning/MyTTSDataset/
```

Or clone fresh on GPU and rsync data:

```bash
# On GPU machine
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
mkdir -p chatterbox-finetuning/MyTTSDataset/wavs

# From local machine
rsync -avP /path/to/fleurs-turkish/train/*.wav user@gpu-host:chatterbox-finetuning/MyTTSDataset/wavs/
scp /path/to/chatterbox-finetuning/MyTTSDataset/metadata.csv user@gpu-host:chatterbox-finetuning/MyTTSDataset/
```

## 4. Train (GPU machine)

```bash
cd chatterbox-finetuning
pip install -r requirements.txt
python setup.py
```

Edit `src/config.py`:

```python
num_epochs: int = 15
learning_rate: float = 5e-6
save_steps: int = 50
save_total_limit: int = 20
ljspeech: bool = True
preprocess: bool = True
is_turbo: bool = False
new_vocab_size: int = 2454
```

```bash
python train.py
```

Output: `chatterbox_output/t3_finetuned.safetensors`

## 5. Test inference (GPU machine)

```bash
cp MyTTSDataset/wavs/1424211997093929997.wav ./reference.wav
python inference.py
```

Edit `inference.py` line 47 to change test text.

### If using a checkpoint instead of final model

Strip the `model.` prefix from checkpoint keys:

```bash
python3 -c "
from safetensors.torch import load_file, save_file
state = load_file('chatterbox_output/checkpoint-XXX/model.safetensors')
clean = {k.replace('model.', '', 1): v for k, v in state.items()}
save_file(clean, 'chatterbox_output/t3_finetuned.safetensors')
"
```

## 6. Deploy to server (GPU machine)

Change `server.py` line 22:

```python
model = ChatterboxTTS.from_local("./models", variant="multilingual")
```

Run container with fine-tuned weights mounted:

```bash
docker run --gpus all -p 4123:4123 \
    -v ~/chatterbox-finetuning/chatterbox_output/t3_finetuned.safetensors:/app/models/t3_mtl23ls_v2.safetensors:ro \
    -v ~/chatterbox-finetuning/chatterbox_output/t3_finetuned.safetensors:/app/t3-model-multilingual/model.safetensors:ro \
    -v ~/chatterbox-finetuning/pretrained_models/ve.safetensors:/app/models/ve.safetensors:ro \
    -v ~/chatterbox-finetuning/pretrained_models/s3gen.safetensors:/app/models/s3gen.safetensors:ro \
    -v ~/chatterbox-finetuning/pretrained_models/conds.pt:/app/models/conds.pt:ro \
    -v ~/chatterbox-finetuning/pretrained_models/grapheme_mtl_merged_expanded_v1.json:/app/models/grapheme_mtl_merged_expanded_v1.json:ro \
    -v ~/chatterbox-finetuning/pretrained_models/Cangjie5_TC.json:/app/models/Cangjie5_TC.json:ro \
    chatterbox-vllm
```

Note: `t3_finetuned.safetensors` is mounted at two paths because `from_local` reads it for conditioning weights and vLLM reads it separately from `t3-model-multilingual/`.

## 7. Disable voice cloning (optional)

Remove entries from `VOICE_CLONE_MAP` in `server.py` to use default model voice instead:

```python
VOICE_CLONE_MAP: dict[str, Path] = {}
```
