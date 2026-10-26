"""SemMSA model: modality encoders, CSR (prompt adapters + frozen LLM) and sum fusion with a regressor."""
import os

import torch
import torch.nn as nn

from .adapter import PromptAdapter
from .bert import BertTextEncoder
from .csr import LatentSemanticRefiner
from .layers import Transformer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
UNK_ID = 100  # BERT [UNK]


class SemMSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        fe, ad, cs = cfg['model']['feature_extractor'], cfg['model']['adapter'], cfg['model']['csr']
        self.T = fe['token_length'][0]
        self.d = fe['hidden_dims'][0]
        self.aux_sem = cfg['train']['loss'].get('w_aux_sem', 0.0) > 0
        # 'all': missing (all-zero) frames and [UNK] tokens are excluded from attention; 'none': attend to every position
        self.mask_missing = cfg['model'].get('mask_missing', 'none')

        # modality encoders -> H_l, H_v, H_a
        self.bertmodel = BertTextEncoder(use_finetune=fe.get('bert_finetune', True), pretrained=fe['bert_pretrained'])
        self.proj_l = self._make_encoder(fe, 0)
        self.proj_v = self._make_encoder(fe, 1)
        self.proj_a = self._make_encoder(fe, 2)

        # CSR: frozen LLM with visual / acoustic prompt adapters
        self.csr = LatentSemanticRefiner(
            llm_path=cs['llm_path'], bert_tokenizer_path=fe['bert_pretrained'], dtype=cs.get('dtype', 'bfloat16'),
            refine_steps=cs.get('refine_steps', 4), chunk_size=cs.get('chunk_size', 16),
            unk_marker=cs.get('unk_marker', '[UNK]'), max_text_tokens=cs.get('max_text_tokens', 96),
            language=cfg['dataset'].get('language', 'en'), state_scale=cs.get('state_scale', 'embed'))
        llm_dim = self.csr.hidden_size
        for name, i in (('adapter_v', 1), ('adapter_a', 2)):
            setattr(self, name, PromptAdapter(fe['input_dims'][i], ad['dim'], llm_dim, ad['num_prompts'], ad['depth'],
                                              ad['heads'], ad.get('mlp_ratio', 4), ad.get('dropout', 0.),
                                              max_len=fe['input_length'][i]))
        # H_s = g_s(Pool(Z)) with a linear or two-layer MLP projection g_s
        if cs.get('sem_proj', 'linear') == 'mlp':
            self.sem_proj = nn.Sequential(nn.Linear(llm_dim, 4 * self.d), nn.GELU(), nn.Linear(4 * self.d, self.d))
        else:
            self.sem_proj = nn.Linear(llm_dim, self.d)

        out_dim = cfg['model']['regression']['out_dim']
        if cfg['model']['regression'].get('type', 'linear') == 'mlp':
            self.classifier = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Linear(self.d, out_dim))
        else:
            self.classifier = nn.Linear(self.d, out_dim)

    def _make_encoder(self, fe, i):
        p = float(fe.get('dropout', 0.0))
        return nn.Sequential(
            nn.Linear(fe['input_dims'][i], fe['hidden_dims'][i]),
            Transformer(num_frames=fe['input_length'][i], token_len=self.T, dim=fe['hidden_dims'][i], depth=fe['depth'],
                        heads=fe['heads'], mlp_dim=fe['hidden_dims'][i], dropout=p, emb_dropout=p))

    @staticmethod
    def _frame_mask(x):
        """True where a frame is all-zero (missing or padding); rows without any observed frame stay unmasked."""
        pad = x.abs().sum(-1) == 0
        return pad & ~pad.all(dim=1, keepdim=True)

    def _encode(self, proj, x, key_padding_mask=None):
        return proj[1](proj[0](x), key_padding_mask=key_padding_mask)[:, :self.T].mean(1)

    def forward(self, vision_m, audio_m, text_m):
        v_pad = a_pad = l_pad = None
        text_bert = text_m
        if self.mask_missing == 'all':
            v_pad, a_pad = self._frame_mask(vision_m), self._frame_mask(audio_m)
            keep_l = text_m[:, 1, :] * (text_m[:, 0, :] != UNK_ID).float()   # [CLS] / [SEP] are never missing
            text_bert = torch.stack((text_m[:, 0, :], keep_l, text_m[:, 2, :]), dim=1)
            l_pad = keep_l == 0

        h_l = self._encode(self.proj_l, self.bertmodel(text_bert), l_pad)
        h_v = self._encode(self.proj_v, vision_m, v_pad)
        h_a = self._encode(self.proj_a, audio_m, a_pad)

        z_v = self.adapter_v(vision_m, key_padding_mask=v_pad)
        z_a = self.adapter_a(audio_m, key_padding_mask=a_pad)
        Z = self.csr(text_m[:, 0, :].long(), z_v, z_a)                     # (B, O, llm_dim)
        h_s = self.sem_proj(Z.mean(1))

        V = torch.stack([h_s, h_v, h_a, h_l], dim=1)                      # (B, 4, d)
        out = {'sentiment_preds': self.classifier(V.sum(1)), 'V': V}
        if self.aux_sem:
            out['aux_preds'] = self.classifier(h_s)                         # semantic branch through the shared regressor
        return out


def build_model(cfg):
    return SemMSA(cfg)
