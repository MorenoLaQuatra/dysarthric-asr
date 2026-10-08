# SAP speaker split

The speaker lists and utterance order used in the paper are not included here. They contain SAP
contributor IDs and file names, which the SAP data-use agreement does not allow us to redistribute.

If you have access to SAP (v31-05-2025) under the data-use agreement, we are happy to share them:
just open an issue or write to us. Place the files in this folder:

- `sap_train_speakers.txt`, `sap_val_speakers.txt`: one contributor ID per line (61 validation speakers)
- `sap_train_val_order.txt.gz`: order of the merged train+validation utterances (Kimi-Audio validation subset)

`data/split_speakers.py` and `data/make_manifests.py` use them when present. Without them, run
`data/split_speakers.py --regenerate` for a new stratified split (seed 42, 10% of the speakers of each
etiology): same procedure, different speakers.
