from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anra_paths import get_dataset_file, get_identity_file, V3_TOKENIZER_FILE, TOKENIZER_DIR
from tokenizer.subword_tokenizer import SubwordTokenizer
SPECIAL_TOKENS=["<unk>","<pad>","<bos>","<eos>","<sep>","<code>","</code>","<think>","</think>","<goal>","<ESV:v>","<ESV:a>","<ESV:d>"]
TARGET_VOCAB=8192

def collect_texts()->list[str]:
    texts=[]
    d=get_dataset_file();
    if d.exists(): texts.append(d.read_text(encoding='utf-8',errors='replace'))
    i=get_identity_file();
    if i.exists(): texts.append(i.read_text(encoding='utf-8',errors='replace'))
    if not texts: texts=["Hello An-Ra\n"*500]
    return texts

def main()->None:
    texts=collect_texts()
    tok=SubwordTokenizer.train_from_texts(texts,vocab_size=TARGET_VOCAB,special_tokens=SPECIAL_TOKENS)
    assert tok.vocab_size==TARGET_VOCAB
    TOKENIZER_DIR.mkdir(parents=True,exist_ok=True)
    tok.save(V3_TOKENIZER_FILE)
    meta=TOKENIZER_DIR/"tokenizer_v3.json.meta.json"
    meta.write_text(json.dumps({"vocab_size":tok.vocab_size,"special_tokens":SPECIAL_TOKENS,"model_type":"bpe","backend":"subword","trained_at":time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},indent=2),encoding='utf-8')
    print(f"saved {V3_TOKENIZER_FILE}")
if __name__=='__main__': main()
