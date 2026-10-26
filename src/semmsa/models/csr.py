"""Cross-modal Semantic Refinement (CSR): latent semantic refinement with a frozen LLM.

The language tokens, the visual prefix Z_v and the acoustic prefix Z_a form the multimodal prefix U^(0).
At every step the last-layer hidden state at the final position, z_k = f(U^(k-1)), is appended to the
sequence as a continuous token, U^(k) = [U^(k-1); z_k], for k = 1..O; Z = [z_1; ...; z_O] is returned.

The prefix pass and the O-1 refinement steps (with the KV cache) of each batch chunk are wrapped in
torch.utils.checkpoint, so gradients reach the adapters without storing the LLM activations.
"""
import re

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoTokenizer, BertTokenizer

SYS_PROMPT = ("You are a multimodal sentiment expert. Fuse the textual semantics with the audio and video "
              "evidence into one sentiment-oriented internal representation. Down-weight missing, noisy or "
              "contradictory evidence and encode sentiment polarity, intensity, attitude and ambiguity into "
              "the final state. Do not explain.")
INST_PROMPT = "Summarize all reliable affective evidence into the final internal state."

_CJK_SPACE = re.compile(r'(?<=[一-鿿　-〿＀-￯])\s+(?=[一-鿿　-〿＀-￯])')
_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}


class LatentSemanticRefiner(nn.Module):
    def __init__(self, llm_path, bert_tokenizer_path, dtype='bfloat16', refine_steps=4, chunk_size=16,
                 unk_marker='[UNK]', max_text_tokens=96, language='en', state_scale='embed'):
        super().__init__()
        self.refine_steps = int(refine_steps)
        self.chunk_size = int(chunk_size)
        self.unk_marker = unk_marker or ''
        self.max_text_tokens = int(max_text_tokens)
        self.language = language
        self.state_scale = state_scale
        self.llm_dtype = _DTYPES[dtype]

        self.tok = AutoTokenizer.from_pretrained(llm_path, padding_side='left')
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.llm = AutoModel.from_pretrained(llm_path, torch_dtype=self.llm_dtype)
        self.llm.config.use_cache = True
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()
        self.hidden_size = self.llm.config.hidden_size
        with torch.no_grad():
            norm = self.llm.get_input_embeddings().weight.float().norm(dim=-1).mean()
        self.register_buffer('embed_norm', norm, persistent=False)

        self.bert_tok = BertTokenizer.from_pretrained(bert_tokenizer_path)
        self.seg_head = self._ids(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\nText: ")
        self.seg_v = self._ids("\nVideo: ")
        self.seg_a = self._ids("\nAudio: ")
        self.seg_tail = self._ids(f"\n{INST_PROMPT}<|im_end|>\n<|im_start|>assistant\n")

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()  # the LLM stays frozen
        return self

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False)['input_ids']

    def _as_token(self, h):
        """Latent state -> next input embedding ('embed': rescaled to the mean token-embedding norm)."""
        if self.state_scale == 'embed':
            h = h.float()
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.embed_norm
        return h.unsqueeze(1).to(self.llm_dtype)

    @torch.no_grad()
    def bert_ids_to_text(self, bert_ids):
        """BERT token ids (missing tokens = [UNK]) -> text for the LLM tokenizer."""
        texts = []
        for row in bert_ids.tolist():
            out = []
            for t in self.bert_tok.convert_ids_to_tokens([int(i) for i in row]):
                if t in ('[CLS]', '[SEP]', '[PAD]'):
                    continue
                if t == '[UNK]':
                    if self.unk_marker:
                        out.append(self.unk_marker)
                    continue
                out.append(t)
            s = self.bert_tok.convert_tokens_to_string(out)
            if self.language == 'cn':
                s = _CJK_SPACE.sub('', s)
            texts.append(s)
        return texts

    def build_inputs(self, bert_ids, z_v, z_a):
        device, b = z_v.device, z_v.size(0)
        text_ids = [self._ids(t)[:self.max_text_tokens] for t in self.bert_ids_to_text(bert_ids)]
        head = [self.seg_head + ids for ids in text_ids]
        max_len = max(len(h) for h in head)
        ids = torch.full((b, max_len), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((b, max_len), dtype=torch.long)
        for i, h in enumerate(head):
            ids[i, max_len - len(h):] = torch.tensor(h, dtype=torch.long)
            mask[i, max_len - len(h):] = 1
        ids, mask = ids.to(device), mask.to(device)
        emb = self.llm.get_input_embeddings()
        with torch.no_grad():
            parts, masks = [emb(ids)], [mask]

        def add_seg(seg):
            t = torch.tensor(seg, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
            with torch.no_grad():
                parts.append(emb(t))
            masks.append(torch.ones(b, t.size(1), dtype=torch.long, device=device))

        def add_prefix(z):
            parts.append(z.to(self.llm_dtype))
            masks.append(torch.ones(b, z.size(1), dtype=torch.long, device=device))

        add_seg(self.seg_v)
        add_prefix(z_v)
        add_seg(self.seg_a)
        add_prefix(z_a)
        add_seg(self.seg_tail)
        embeds, attn = torch.cat(parts, dim=1), torch.cat(masks, dim=1)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        return embeds, attn, pos

    def _chunk_refine(self, embeds, attn, pos):
        """Prefix pass + (O-1) KV-cached refinement steps for one chunk -> (b, O, hidden)."""
        b = embeds.size(0)
        out = self.llm(inputs_embeds=embeds, attention_mask=attn, position_ids=pos, use_cache=True, return_dict=True)
        cache, states, cur_m = out.past_key_values, [out.last_hidden_state[:, -1]], attn
        n_real = attn.sum(1, keepdim=True)
        for k in range(1, self.refine_steps):
            cur_m = torch.cat([cur_m, torch.ones(b, 1, dtype=cur_m.dtype, device=cur_m.device)], dim=1)
            out = self.llm(inputs_embeds=self._as_token(states[-1]), attention_mask=cur_m, position_ids=n_real + (k - 1),
                           past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            states.append(out.last_hidden_state[:, -1])
        return torch.stack(states, dim=1)

    def forward(self, bert_ids, z_v, z_a):
        embeds, attn, pos = self.build_inputs(bert_ids, z_v, z_a)
        b = embeds.size(0)
        need_grad = torch.is_grad_enabled() and embeds.requires_grad
        chunk = self.chunk_size if (need_grad and self.chunk_size > 0) else b
        outs = []
        for s in range(0, b, chunk):
            e, m, p = embeds[s:s + chunk], attn[s:s + chunk], pos[s:s + chunk]
            outs.append(checkpoint(self._chunk_refine, e, m, p, use_reentrant=True) if need_grad else self._chunk_refine(e, m, p))
        return torch.cat(outs, dim=0).float()
