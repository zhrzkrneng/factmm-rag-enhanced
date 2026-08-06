# Milestone 2.8 Cell 40 -- Download Plan

**This is a plan, not an execution.** No download occurs as a result of this file. Every step below is manual/user-initiated, per `docs/milestone_2_8_resource_acquisition_contract.md` SS5/SS6, unchanged by this cell.

## Order of acquisition (mandatory, blocking resources first)

1. **mimic_cxr** (dataset) -- source: PhysioNet, processed per Delbrouck et al. 2023 (vilmedic ACL-2023 RadSum23 split); URL: https://vilmedic.app/papers/acl2023/; auth: PhysioNet account + CITI training + signed Data Use Agreement; destination: data/mimic/
2. **chexpert** (dataset) -- source: Stanford AIMI Shared Datasets; URL: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2; auth: Stanford AIMI account registration; destination: data/chexpert/
3. **marvel_warm_start_checkpoint** (checkpoint) -- source: Hugging Face Hub; URL: https://huggingface.co/OpenMatch/marvel-ance-clueweb/tree/main; auth: none known (public repo); destination: ~/.cache/huggingface/hub/models--OpenMatch--marvel-ance-clueweb/ (or a project-local path mirroring the official repo's ./src/checkpoint/ convention if loaded via --pretrained_model_path)
4. **llava_base_vicuna** (checkpoint) -- source: Hugging Face Hub; URL: https://huggingface.co/lmsys/vicuna-7b-v1.5; auth: UNKNOWN; destination: UNKNOWN
5. **llava_fork_haotian_liu** (code_dependency) -- source: GitHub; URL: https://github.com/haotian-liu/LLaVA.git; auth: none (public GitHub repo); destination: FactMM-RAG/LLaVA/ (per install_llava.sh's own cd sequence)
6. **llava_generator_vision_tower_clip_l14** (checkpoint) -- source: Hugging Face Hub; URL: https://huggingface.co/openai/clip-vit-large-patch14-336; auth: UNKNOWN; destination: UNKNOWN
7. **llava_mm_projector_checkpoint** (checkpoint) -- source: UNKNOWN; URL: UNKNOWN; auth: UNKNOWN; destination: UNKNOWN

## Already available / non-blocking resources

- **clip_vit_b32_retriever**: Default in DPR/train.py's --clip_model_name; real-verified, Milestone 2.4G; AVAILABLE in at least one checked environment (Cell 38).
- **t5_ance**: Default in DPR/train.py's --t5_model_name; real-verified, Milestone 2.4G; AVAILABLE in at least one checked environment (Cell 38).
- **unlabeled_rag_section_checkpoint**: NEWLY DISCOVERED this run -- README.md line 84, an unlabeled 'Checkpoint:' Google Drive link appearing at the very start of the RAG section, before the knn.py commands. Its exact identity (a pretrained DPR/ANCE retriever checkpoint is the most plausible reading given its position, but this is REASONABLE_INFERENCE, not confirmed) is UNKNOWN. Disclosed as found, not resolved by guessing.

## Per-resource acquisition method

- **Datasets (MIMIC-CXR, CheXpert)**: user completes credentialing at the official source, downloads manually, places files at the destination path, then a future milestone's `ManifestBuilder` run (Milestone 2.1, reused unmodified) generates the manifest. This project's own tooling never automates the credentialing or download step itself.
- **Hugging Face Hub checkpoints (MARVEL, CLIP variants, T5-ANCE, vicuna)**: standard `transformers`/`huggingface_hub` resolution, pinned to the exact revision recorded in `resource_inventory.json` wherever known; `UNKNOWN` revisions must be resolved (by pinning one explicitly) before acquisition, not downloaded as unpinned 'latest'.
- **LLaVA fork (`haotian-liu/LLaVA`)**: `git clone` at the now-known pinned commit `c121f0432da27facab705978f83c4ada465e46fd`, installed per `install_llava.sh`'s own documented steps.
- **LLaVA MM projector checkpoint**: source unresolved (`UNKNOWN`) -- acquisition cannot be planned until its identity is determined (open question, not solvable from any source read this session).
- **Unlabeled RAG-section checkpoint**: optional, identity ambiguous -- acquisition deferred until its exact purpose is confirmed, since downloading an unidentified artifact would violate this project's never-fabricate/never-guess discipline.

## Explicit non-actions

This plan does not download, does not fetch, does not clone, and does not verify any resource beyond what Cells 37-39 already did (offline, no-network-call checks). Executing this plan is deferred to a future, separately-authorized implementation milestone.
