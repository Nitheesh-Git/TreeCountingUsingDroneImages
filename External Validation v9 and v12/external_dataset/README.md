# External dataset (held out)

Reserved for an independent drone dataset used only for a final generalisation
check of the already-trained models.

Rules:
- Never used for training, validation, early stopping, checkpoint selection or
  hyperparameter decisions.
- Everything in this folder except this README is git-ignored; record the
  source, licence and image count here when a dataset is added.

Status: `testdataset100/` (SelvaBox, 100 tiles from the official test split, 8,111 crowns, CC-BY-4.0,
prepared by Astley) is placed here locally and is git-ignored. Both models were evaluated on it zero-shot
with `python "External Validation v9 and v12/common/evaluate_external.py`;" see `../REPORT.md` section 9.
