# J-space on Llama 3.1 8B Instruct

This project will test a minimal causal J-space intervention on
`meta-llama/Llama-3.1-8B-Instruct` using the prefitted lens at
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`.

The intended workflow runs on a Colab NVIDIA L4 with gated Hugging Face model
access, downloads model and lens assets only into ignored caches, and retains
only compact aggregate JSON results. The planned causal check swaps the
intermediate concept `spider` for `ant` and compares the resulting answer with
baseline and control interventions.

The reusable core now provides pinned configuration, CUDA/runtime validation,
memory-bounded readout and interventions, controls, and structured observation
orchestration. `notebooks/01_readout_and_causal_swap.ipynb` is the thin entry
point for the real run. Results are not yet available; the 8B model is
intentionally not executed in the repository test suite.
