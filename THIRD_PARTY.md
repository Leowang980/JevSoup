# Third-Party Resources

Model weights, datasets and hosted services are obtained separately and retain
their own licenses and terms. This repository does not redistribute them.

| Resource | Use | Source |
| --- | --- | --- |
| PorTAL / portallib | Task data, expert adapters and continuation-scoring convention | `portallib==0.2.1`; [PorTAL tasks](https://huggingface.co/datasets/RampPublic/portallib-tasks) |
| Qwen3 | Frozen 1.7B, 4B and 8B backbones | [Qwen](https://huggingface.co/Qwen); revisions in `jev_lora/core.py` |
| Qwen3-Embedding-0.6B | AdapterSoup training-sample retrieval | revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Jev / TypeSafe | Hosted System One routing | [TypeSafe](https://typesafe.ai/); experiment model `jev-1.13.0` |
| MTTL / Arrow | Weight-derived prototypes and token-level routing | [Microsoft MTTL](https://github.com/microsoft/mttl), commit `169c9191be960e35a59e85c37af90f3f518fe125` |
| LoGo | Activation-based expert selection | [LoGo](https://github.com/archon159/LoGo) |
| Adaptive Minds | Generative expert selection | [Adaptive Minds](https://github.com/qpiai/adaptive-minds) |
| AdapterSoup | Training-sample-based selection and weight averaging | Chronopoulou et al., *AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models* |

The MIT notice for adapted MTTL routines is preserved in
[`third_party/mttl-LICENSE`](third_party/mttl-LICENSE). The local baselines adapt
the original methods to a common Qwen3 expert pool and multiple-choice protocol;
see [docs/PROTOCOL.md](docs/PROTOCOL.md).

The project license does not replace upstream model, dataset or service terms.
