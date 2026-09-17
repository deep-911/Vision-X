# AI Agent Directives for VisionX (SIH26038)

This file defines the high-level directives for AI coding assistants working on the VisionX project.

1. **Architecture Preservation**: Never reduce or simplify the intended SIH26038 architecture (as defined in `SIH26038_Idea.md`) to accommodate the local hardware (RTX 3050 4GB).
2. **Environment Awareness**: Acknowledge that local training capability is limited. Design ML pipelines such that heavy training occurs in Google Colab, but the codebase remains unified.
3. **MATLAB as a First-Class Citizen**: Ensure all ML components ultimately integrate with MATLAB/Simulink. Use ONNX as the primary interoperability boundary.
4. **Data Privacy**: Never commit datasets, API keys, or large artifacts to the repository.
5. **Scientific Integrity**: Do not invent metrics, fake validation, or hallucinate results.
