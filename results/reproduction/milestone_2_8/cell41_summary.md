# Milestone 2.8 Cell 41 -- Resource Acquisition Planning Summary

Built strictly on Cell 40's `resource_inventory.json` (10 resources, unmodified) -- classification never re-investigates or guesses beyond what Cell 40 already recorded.

## Category counts

- Category A (Public / Automatically downloadable): 4
- Category B (Credential required): 2
- Category C (Ambiguous or unresolved): 4
- Total: 10 (matches Cell 40's 10-resource inventory exactly)

| Resource | Category | Manual/Automatic | User interaction required | Cell 42 auto-acquirable |
|---|---|---|---|---|
| chexpert | B_credential_required | manual | True | False |
| clip_vit_b32_retriever | A_public_automatic | automatic | False | True |
| llava_base_vicuna | C_ambiguous_unresolved | UNKNOWN | True | False |
| llava_fork_haotian_liu | A_public_automatic | automatic | False | True |
| llava_generator_vision_tower_clip_l14 | C_ambiguous_unresolved | UNKNOWN | True | False |
| llava_mm_projector_checkpoint | C_ambiguous_unresolved | UNKNOWN | True | False |
| marvel_warm_start_checkpoint | A_public_automatic | automatic | False | True |
| mimic_cxr | B_credential_required | manual | True | False |
| t5_ance | A_public_automatic | automatic | False | True |
| unlabeled_rag_section_checkpoint | C_ambiguous_unresolved | UNKNOWN | True | False |

## Readiness for a future Cell 42

4 of 10 resources are classified Category A and could, in principle, be acquired automatically by a future, separately-authorized Cell 42 -- but only for the subset where a revision/commit is already pinned (4 resources meet that bar). No Category B or C resource can be acquired without a human action first (credentialing, or resolving an open identity question).

## Scope discipline

No dataset or checkpoint was downloaded. No baseline source file or test was modified. This cell classifies and plans only.
