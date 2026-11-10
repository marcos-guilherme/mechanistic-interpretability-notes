# Llama J-space causal milestone

## Goal and causal-minimum scope

This project will establish one narrow technical milestone: determine whether a
prefitted Jacobian lens and the pinned `jspace` implementation can expose and
causally reroute one intermediate concept in
`meta-llama/Llama-3.1-8B-Instruct`.

The minimum experiment uses the prompt “The animal that spins webs has this
many legs:” and checks three linked observations:

1. the unmodified model produces the expected spider-associated answer,
   normally “eight”;
2. the J-space readout surfaces `spider` in the model's mid-network workspace
   band;
3. swapping `spider` to `ant` changes the downstream answer toward the
   ant-associated answer, normally “six”.

The experiment uses the prefitted lens
`llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt`
and the `jspace` repository pinned at commit
`04c822a255fe0a10d84e08ec499e5e09f4b7d37a`. It does not fit a new lens.

## Controls

The causal comparison must record the same prompt and generation settings for:

- an unmodified baseline;
- a no-op `spider -> spider` intervention;
- a norm-matched random-direction intervention;
- the target `spider -> ant` swap.

The baseline establishes that the model can solve the prompt before
intervention. The no-op detects hook or reconstruction side effects. The
norm-matched random direction tests whether an arbitrary perturbation has the
same effect. The target swap is persuasive only when it changes the downstream
fact without merely emitting `ant` or degrading the continuation.

## Inconclusive conditions

The result is inconclusive if the baseline does not answer with the
spider-associated fact, the lens does not surface `spider`, the target swap
does not favor the ant-associated fact, or either control produces a comparable
rerouting. It is also inconclusive if the intervention leaks the target entity,
causes repetition or incoherent output, depends on an undocumented prompt or
generation change, cannot be repeated under the recorded seed and settings, or
lacks enough metadata to distinguish intervention failure from environment or
asset failure. An inconclusive run must not be reported as negative evidence
against J-space.

## Colab L4 execution

Execution is intended for one Google Colab runtime with an NVIDIA L4. The
runtime will use Python 3.11–3.13, PyTorch in `bfloat16`, authenticated gated
access to the Llama model, the pinned project environment, and the exact
prefitted lens path above. Later tasks may add thin notebooks or scripts for
setup and execution; this skeleton does not implement model loading,
interventions, Colab orchestration, or result validation.

The run must record the model ID and revision, lens path and hash, `jspace`
commit, Python and dependency versions, GPU, dtype, seed, prompt, generation
settings, intervention parameters, readout layers, and local Git commit.

## Artifact policy

Model weights, Hugging Face caches, the prefitted lens, raw or heavy tensors,
raw generations, credentials, local environments, and generated notebook
outputs remain outside Git. Compact aggregate `results/*.json` files may be
versioned when they contain metrics and provenance without secrets or model
assets. A result must identify any local-only raw artifact by hash rather than
copying it into the repository.

## Limitation

Success would demonstrate that this particular software, model, lens, prompt,
and intervention can produce one controlled causal rerouting. It is a
technical milestone, not a general replication of the paper: it does not test
the paper's datasets, breadth of models, lens fitting, reportability metrics,
workspace geometry, ablations, behavioral generality, or statistical claims.
