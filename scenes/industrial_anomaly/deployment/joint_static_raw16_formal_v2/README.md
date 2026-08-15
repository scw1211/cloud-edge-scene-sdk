# Joint static Q4 raw16 formal v2

This Nano222 deployment serves one statically fused Q4_K_M model for both
traffic and industrial decisions. Both scene encoders emit their existing
16-digit task code unchanged. The traffic and industrial domains are disjoint
in the frozen datasets, so the model needs no extra scene-prefix token and no
runtime LoRA selection.

Runtime invariants:

- input contract: `scene-disjoint-decimal16@v2`;
- output contract: exactly one constrained action token;
- runtime adapters: `[]`;
- llama.cpp: ctx 128, batch/ubatch 16, parallel 1, all GPU layers;
- static model SHA-256: `828f839873c7505005101544d8febb7408bc0443c153c4ba97ec4b3603445526`.

The formal unit uses an isolated registry and storage directory. The prior
joint-static Q5 unit is its rollback target and is never overwritten.
