# Machine Learning Rules

- **Hardware Independence & Architecture**: Preserve the full SIH26038 architecture regardless of local hardware constraints. Use the local RTX 3050 for development and small tests, and Colab for heavy training.
- **Colab as Wrapper**: Colab notebooks must be thin wrappers that clone the repository. The core ML implementation must live in the repo scripts, not duplicated in notebooks.
- **Script Robustness**: Training scripts must support dynamic device selection (CUDA/CPU fallback), mixed precision, and resume-from-checkpoint functionality.
- **Reproducible Tracking**: Every experiment must record run ID, git commit, dataset manifest version, configuration, model, metrics, checkpoint path, and timestamp.
- **Colab Checkpointing**: All heavy training must save regular checkpoints, configurations, metrics, and separately save the best checkpoint.
- **Baselines**: Start with lightweight CNN + standard classification vs. lightweight CNN + CORAL ordinal head.
- **Quality Interface**: The MVP quality module must be implemented behind a stable interface to allow swapping with the advanced model later.
