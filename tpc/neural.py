"""Optional neural scorers (need `torch` and `transformers`; model weights are downloaded once from the Hugging Face Hub
and cached locally, after which everything runs offline). Default models fit easily on an 8 GB GPU (or run on CPU).

  ppl     mean token negative log-likelihood under a small causal LM     (default: gpt2, 124M)
  pi_clf  INJECTION-vs-SAFE logit of a public prompt-injection classifier (default: protectai/deberta-v3-base-prompt-injection-v2)
  emb     L2-normalised sentence embeddings (CLS pooling) for emb_knn / emb_lr (default: BAAI/bge-small-en-v1.5)

Any local folder holding a compatible model can be passed instead of a Hub name (e.g. for an offline machine).
"""
import numpy as np

DEFAULTS = dict(ppl='gpt2', pi='protectai/deberta-v3-base-prompt-injection-v2', emb='BAAI/bge-small-en-v1.5')


def _torch(device):
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch, device


def _batches(n, bs):
    for i in range(0, n, bs):
        yield slice(i, min(n, i + bs))


def perplexity(texts, model_name=DEFAULTS['ppl'], device='auto', batch_size=16, max_len=256):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch, device = _torch(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    tok.padding_side = 'right'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    bos = tok.bos_token or tok.eos_token or ''
    out = np.empty(len(texts))
    with torch.no_grad():
        for sl in _batches(len(texts), batch_size):
            enc = tok([bos + t for t in texts[sl]], return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            logits = model(**enc).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            tgt = enc['input_ids'][:, 1:]; mask = enc['attention_mask'][:, 1:].float()
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask
            out[sl] = (nll.sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
    return out


def injection_logit(texts, model_name=DEFAULTS['pi'], device='auto', batch_size=16, max_len=512):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch, device = _torch(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    lab = {v.upper(): int(k) for k, v in model.config.id2label.items()}
    pos = lab.get('INJECTION', lab.get('LABEL_1', model.config.num_labels - 1))
    neg = lab.get('SAFE', lab.get('BENIGN', lab.get('LABEL_0', 0)))
    out = np.empty(len(texts))
    with torch.no_grad():
        for sl in _batches(len(texts), batch_size):
            enc = tok(list(texts[sl]), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            lg = model(**enc).logits.float()
            out[sl] = (lg[:, pos] - lg[:, neg]).cpu().numpy()
    return out


def embed(texts, model_name=DEFAULTS['emb'], device='auto', batch_size=32, max_len=512):
    from transformers import AutoModel, AutoTokenizer
    torch, device = _torch(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for sl in _batches(len(texts), batch_size):
            enc = tok(list(texts[sl]), return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(device)
            h = model(**enc).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.vstack(out).astype(np.float32)
