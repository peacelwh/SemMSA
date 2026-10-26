# Data

SemMSA is evaluated on three multimodal sentiment analysis benchmarks. We use the processed (unaligned) features
released with [MMSA](https://github.com/thuiar/MMSA), which are also used by LNLN and P-RMF.

```bash
bash scripts/download_data.sh            # all three datasets
bash scripts/download_data.sh sims       # a single dataset
```

| File | Dataset | Speakers | Clips | Train | Valid | Test | Language | Size |
|---|---|---|---|---|---|---|---|---|
| `mosi_unaligned_50.pkl` | CMU-MOSI | 93 | 2,199 | 1,284 | 229 | 686 | English | 0.6 GB |
| `mosei_unaligned_50.pkl` | CMU-MOSEI | 1,000 | 22,856 | 16,326 | 1,871 | 4,659 | English | 13.7 GB |
| `sims_unaligned_39.pkl` | CH-SIMS | 474 | 2,281 | 1,368 | 456 | 457 | Chinese | 1.2 GB |

CMU-MOSEI is stored on the Hub as 1 GiB parts (`mosei_unaligned_50/mosei_unaligned_50.pkl.part00`-`part12`);
`download_data.sh` reassembles them into `mosei_unaligned_50.pkl` and checks its SHA-256.

Each pickle holds `train` / `valid` / `test` splits with BERT token ids (`text_bert`), visual features (`vision`),
acoustic features (`audio`), sequence lengths and sentiment labels (`regression_labels`; MOSI / MOSEI in [-3, 3],
SIMS in [-1, 1]).

The datasets are redistributed for research purposes only and remain subject to the terms of their original releases.
If you use them, please cite the original papers:

- CMU-MOSI: Zadeh et al., *Multimodal Sentiment Intensity Analysis in Videos: Facial Gestures and Verbal Messages*, IEEE Intelligent Systems, 2016.
- CMU-MOSEI: Zadeh et al., *Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph*, ACL 2018.
- CH-SIMS: Yu et al., *CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality*, ACL 2020.
