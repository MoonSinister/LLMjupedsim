# Reproduction Guide

1. Create Python 3.12 environment and install environment/requirements.txt.
2. Set PYTHONPATH=src in the source checkout.
3. Run python -m jupedsim_mall quality-gate.
4. Prepare the frozen matrix under configs/matrices.
5. Execute or resume the matrix; never delete failed attempts.
6. Run analyze metrics, analyze realism, analyze statistics, analyze states.
7. Run report build and compare hashes in this release.

Restricted ATC/LLMob raw data are intentionally excluded. The included smoke matrix requires no remote LLM or restricted data.
