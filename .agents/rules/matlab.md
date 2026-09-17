# MATLAB and Simulink Rules

- **First-Class Integration**: MATLAB/Simulink are first-class components, as required by the SIH26038 prompt.
- **Early Interoperability**: Establish Python -> ONNX -> MATLAB interoperability during the MVP phase, not at the end. Prove export and consumption early.
- **Scriptability**: Prefer scriptable and reproducible MATLAB/Simulink generation over manual GUI-only steps. Ensure batch execution works (`matlab -batch`).
- **Do not modify** or open the existing MATLAB project until the base models are built and ready for integration.
