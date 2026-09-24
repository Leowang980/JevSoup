# Contributing

Discuss substantial changes before changing the evaluation protocol. Include a
small regression test with a bug report or pull request, and state the Python,
PyTorch, Transformers, PEFT and GPU versions used.

Run `python -m unittest discover -s tests -v`. Numerical kernels are deliberately
frozen: editing `jev_lora/*.py` changes cache identity. Changes to scoring,
projection, expert ordering or routes require new output directories and new
equivalence evidence. Do not overwrite existing predictions or reuse their
metadata to disguise a protocol change.

Never commit keys, `.env` files, raw API payloads, local model weights or data
downloads. Describe security problems without posting credentials or private
inputs. The repository owner will add a private contact channel before release.
