# Milestone 2.8 Cell 41 -- Manual Acquisition Steps

Every resource below requires human action before acquisition can proceed -- either because it is credential-gated (Category B) or because its identity/requirements are still unresolved (Category C). Category A resources are listed separately (automatic, no human action beyond authorizing a future Cell 42 to run).

## Category B -- Credential required

### chexpert
- Source: Stanford AIMI Shared Datasets
- URL: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
- Required authentication: Stanford AIMI account registration
- Destination: data/chexpert/
- Action: manual, credentialed download by the user from Stanford AIMI Shared Datasets (https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2); this project's tooling never automates this step

### mimic_cxr
- Source: PhysioNet, processed per Delbrouck et al. 2023 (vilmedic ACL-2023 RadSum23 split)
- URL: https://vilmedic.app/papers/acl2023/
- Required authentication: PhysioNet account + CITI training + signed Data Use Agreement
- Destination: data/mimic/
- Action: manual, credentialed download by the user from PhysioNet, processed per Delbrouck et al. 2023 (vilmedic ACL-2023 RadSum23 split) (https://vilmedic.app/papers/acl2023/); this project's tooling never automates this step

## Category C -- Ambiguous or unresolved (must be resolved before any acquisition plan can be made)

### llava_base_vicuna
- What is known: NEWLY RESOLVED this run: exact identifier lmsys/vicuna-7b-v1.5, confirmed directly in src/generator/train_llava.sh's --model_name_or_path argument -- previously UNKNOWN in every prior milestone's docs (Cell 37, Milestone 2.7/2.8 contracts).
- What is missing: authentication_requirement=UNKNOWN, official_source=Hugging Face Hub
- Action: further investigation required (not a download step) before this resource can be classified A or B.

### llava_generator_vision_tower_clip_l14
- What is known: NEWLY DISCOVERED this run -- not previously tracked anywhere in this project. Distinct from the retriever's own openai/clip-vit-base-patch32 vision encoder; LLaVA's own generator stage uses this larger, different CLIP variant as its vision tower (train_llava.sh and vqa/train_llava_vqa.sh both confirm this, identically).
- What is missing: authentication_requirement=UNKNOWN, official_source=Hugging Face Hub
- Action: further investigation required (not a download step) before this resource can be classified A or B.

### llava_mm_projector_checkpoint
- What is known: NEWLY DISCOVERED this run -- README's RAG section sets an environment variable PROJECTOR_PATH="path_to_llava_projector" (line 113) consumed by train_llava.sh's --pretrain_mm_mlp_adapter $PROJECTOR_PATH (line 11). Neither the README nor any script read this run names a specific source/repo for this checkpoint -- its exact identity is UNKNOWN, not guessed. Likely a standard LLaVA pretrain-stage MM projector artifact by convention, but that is REASONABLE_INFERENCE only, not confirmed by any source read this session, and is not recorded as fact here.
- What is missing: authentication_requirement=UNKNOWN, official_source=UNKNOWN
- Action: further investigation required (not a download step) before this resource can be classified A or B.

### unlabeled_rag_section_checkpoint
- What is known: NEWLY DISCOVERED this run -- README.md line 84, an unlabeled 'Checkpoint:' Google Drive link appearing at the very start of the RAG section, before the knn.py commands. Its exact identity (a pretrained DPR/ANCE retriever checkpoint is the most plausible reading given its position, but this is REASONABLE_INFERENCE, not confirmed) is UNKNOWN. Disclosed as found, not resolved by guessing.
- What is missing: authentication_requirement=UNKNOWN, official_source=Google Drive (link only, no further description in any file read this session)
- Action: further investigation required (not a download step) before this resource can be classified A or B.

## Category A -- Public / automatic (for reference; still requires a future, separately-authorized Cell 42 to execute)

- **clip_vit_b32_retriever**: Hugging Face Hub download (Hugging Face Hub), pinned to revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
- **llava_fork_haotian_liu**: git clone (https://github.com/haotian-liu/LLaVA.git) at pinned commit c121f0432da27facab705978f83c4ada465e46fd
- **marvel_warm_start_checkpoint**: Hugging Face Hub download (Hugging Face Hub), pinned to revision 19bd4191e36a285ffa13cad901c670cd785a4aec
- **t5_ance**: Hugging Face Hub download (Hugging Face Hub), pinned to revision bf70ee32b49c3e8c1d40982feebbc3b9930eeab4

## Scope discipline

This file lists what a human (or a future, separately-authorized Cell 42) would need to do. No action listed here has been performed by this cell. No dataset or checkpoint was downloaded.
