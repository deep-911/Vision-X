# SIH26038 --- Evidence-Aware Retinal Intelligence

## Final Project Idea & Implementation Specification

## 1. Project Vision

Build a reliability-aware, ordinal, explainable and offline-first AI
screening system for diabetic retinopathy (DR) in rural India.

The system should not behave like a simple image classifier. It should
determine:

-   whether a fundus image is good enough to analyze,
-   the DR severity from Grade 0 to Grade 4,
-   which retinal lesions provide evidence for that assessment,
-   whether independent evidence sources agree,
-   how trustworthy the prediction is,
-   whether the image is outside the model's familiar distribution,
-   and whether the case should be automatically reported, sent for
    human review, or recaptured.

The central engineering principle is:

> **The AI should know when its evidence is trustworthy and should be
> able to abstain when it is not.**

The system is an assistive screening system. Final clinical decisions
remain with an ophthalmologist.

------------------------------------------------------------------------

# 2. Core System Architecture

``` text
                         PATIENT
                            │
                            ▼
                  Portable Fundus Camera
                            │
                            ▼
                 ┌─────────────────────┐
                 │   IMAGE QUALITY AI  │
                 └─────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
            GOOD        BORDERLINE        BAD
              │             │             │
              │       Enhancement        Recapture
              │             │
              └─────────────┴──────────────┐
                                           ▼
                              RETINAL STANDARDIZATION
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                          ▼                                 ▼
                  RETINAL STRUCTURE                 LESION ANALYSIS
                          │                                 │
                  ┌───────┼────────┐              ┌────────┼────────┐
                  │       │        │              │        │        │
                Vessel    OD     Fovea            MA       EX       HE
                                                     │        │        │
                                                     └────────┼────────┘
                                                              ▼
                                                   LESION EVIDENCE VECTOR
                                                              │
                     ┌────────────────────────────────────────┴──────────────────┐
                     │                                                           │
                     ▼                                                           ▼
              ORDINAL DR MODEL                                           STRUCTURED MODEL
              CORAL / ordinal                                               Radiomics
              Transformer/CNN                                                + TabPFN
                     │                                                           │
                     └──────────────────────────┬────────────────────────────────┘
                                                ▼
                                      EVIDENCE FUSION ENGINE
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                         ▼                      ▼                      ▼
                    Prediction             Confidence             Evidence
                    Grade 0–4               calibrated             agreement
                         │                      │                      │
                         └──────────────────────┼──────────────────────┘
                                                ▼
                                        TRUST / RISK ENGINE
                                                │
                         ┌──────────────────────┼──────────────────────┐
                         │                      │                      │
                         ▼                      ▼                      ▼
                      TRUSTED               UNCERTAIN              HIGH RISK
                         │                      │                      │
                         ▼                      ▼                      ▼
                    AI-assisted          Human review          Referral workflow
                       result                  │                      │
                                               └──────────┬───────────┘
                                                          ▼
                                                   Ophthalmologist
                                                          │
                                                          ▼
                                                   Final decision
                                                          │
                                                          ▼
                                              Longitudinal patient record
                                                          │
                                                          ▼
                                                 District analytics
                                                          │
                                                          ▼
                                                  MATLAB / Simulink
```

------------------------------------------------------------------------

# 3. Image Quality Intelligence

Image quality should be a learned clinical component rather than a
single blur threshold.

The quality model should evaluate:

-   focus/sharpness,
-   illumination,
-   field of view,
-   retinal coverage,
-   anatomical visibility,
-   artifacts,
-   exposure,
-   glare,
-   image centering,
-   usable retinal area.

The output should be a continuous quality score plus interpretable
sub-scores.

``` text
                         FUNDUS IMAGE
                              │
                  ┌───────────┴───────────┐
                  │                       │
              Quality AI            Quality Features
                  │                       │
                  └───────────┬───────────┘
                              ▼
                         Quality Score
                              │
             ┌────────────────┼────────────────┐
             │                │                │
            GOOD          BORDERLINE          BAD
             │                │                │
             ▼                ▼                ▼
          Analyze        Enhancement         Recapture
```

Example interface:

``` text
IMAGE QUALITY

Focus              94%
Illumination        89%
Retinal coverage    96%
Artifact score       3%
Anatomical visibility 91%

QUALITY: ACCEPTED
```

A borderline image should first pass through enhancement and be
reassessed. A clearly unusable image should trigger recapture rather
than a forced diagnosis.

------------------------------------------------------------------------

# 4. Retinal Standardization

Before disease inference, standardize the image.

Pipeline:

``` text
Input
  ↓
Color normalization
  ↓
Retinal field detection
  ↓
Crop / resize
  ↓
Illumination correction
  ↓
Optional enhancement
  ↓
Standardized retinal image
```

Training should include realistic acquisition variations:

-   blur,
-   motion blur,
-   JPEG compression,
-   low resolution,
-   uneven illumination,
-   overexposure,
-   underexposure,
-   glare,
-   dust/lens artifacts,
-   small field of view,
-   off-center retina,
-   vignette,
-   contrast variation.

The goal is robustness to the conditions expected from
resource-constrained and portable fundus cameras.

------------------------------------------------------------------------

# 5. Retinal Representation Model

Do not rely on training a conventional CNN from scratch as the main
representation.

Preferred direction:

``` text
Retinal foundation model
        ↓
Fine-tuning
        ↓
Ordinal DR head
```

A practical alternative for efficient deployment:

``` text
ConvNeXt-Tiny / EfficientNet
        +
CORAL ordinal head
```

RETFound and subsequent retinal foundation-model research support the
use of large-scale retinal pretraining for downstream ophthalmic tasks.

Model selection must ultimately be based on validation performance,
calibration, robustness, inference latency and deployment constraints
rather than model size alone.

------------------------------------------------------------------------

# 6. Ordinal Diabetic Retinopathy Grading

DR severity is ordered:

``` text
0 → 1 → 2 → 3 → 4
```

The main DR model should therefore use an ordinal formulation rather
than treating the five grades as unrelated classes.

Use:

``` text
P(Grade > 0)
P(Grade > 1)
P(Grade > 2)
P(Grade > 3)
```

Example:

``` text
P(DR > 0) = 0.94
P(DR > 1) = 0.81
P(DR > 2) = 0.37
P(DR > 3) = 0.04
```

This provides a natural representation of severity progression.

For referable DR screening:

``` text
P(Referable DR) = P(Grade ≥ 2)
```

The exact clinical referral threshold should follow the problem
statement and the clinical definition adopted by the project.

Use a CORAL-style ordinal formulation as the primary implementation
candidate.

------------------------------------------------------------------------

# 7. Lesion-Level Evidence

The system should not depend entirely on the global classifier.

Build a lesion-analysis branch capable of identifying evidence such as:

-   microaneurysms (MA),
-   exudates (EX),
-   hemorrhages (HE),
-   other relevant retinal abnormalities where sufficient training data
    exists.

Recommended segmentation model:

``` text
SegFormer-B2 / SegFormer-B3
```

Use segmentation as an evidence generator.

Do not claim that lesion segmentation alone establishes the clinical
diagnosis.

The output should include:

``` text
Lesion Evidence Vector

Microaneurysm count
Microaneurysm area
Exudate count
Exudate area
Hemorrhage count
Hemorrhage area
Lesion density
Macular-region involvement
Spatial distribution
```

------------------------------------------------------------------------

# 8. Independent Lesion Verification

Use multiple independent signals.

``` text
                    Fundus Image
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      DR model       SegFormer      Morphological
      attention       lesions         detector
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Evidence agreement
```

Example:

``` text
SegFormer:
Exudate probability = 0.91

Morphological analysis:
Exudate candidate = YES

Classifier attention:
Overlaps lesion region

→ Evidence agreement = HIGH
```

If the global model predicts severe disease but lesion evidence is
absent:

``` text
Classifier:
Severe DR

Segmentation:
No significant lesions

Morphology:
No significant lesions

→ Evidence disagreement
→ HUMAN REVIEW
```

This is more meaningful than displaying an explainability heatmap
without validation.

------------------------------------------------------------------------

# 9. Evidence-Based Explainability

The system should produce an explanation based on measurable evidence.

Example:

``` text
PREDICTION

Grade 2 — Moderate NPDR


EVIDENCE

Microaneurysms       Detected
Exudates             Detected
Hemorrhages          Low
Vascular abnormality Detected


SPATIAL EVIDENCE

MA candidate regions: 14
EX candidate regions: 4
HE candidate regions: 1


CONSISTENCY

Grad-CAM ↔ lesion overlap: 0.73
Segmentation ↔ morphology: 0.81


CONFIDENCE

P(Grade ≥ 2): 0.91
Calibrated confidence: 0.87


ACTION

REFER TO OPHTHALMOLOGIST
```

The objective is to answer:

1.  What did the model predict?
2.  Why did it predict it?
3.  Where is the supporting evidence?
4.  Do independent evidence sources agree?
5.  How trustworthy is the prediction?
6.  What action should the workflow take?

------------------------------------------------------------------------

# 10. Trust Score

Create a system-level trust score that combines independent reliability
signals.

Candidate formulation:

``` text
Trust Score =
    w1 × Image Quality
  + w2 × Model Confidence
  + w3 × Evidence Agreement
  + w4 × Lesion-Attribution Agreement
  + w5 × Calibration Reliability
  + w6 × OOD Reliability
```

The weights must be learned/tuned experimentally on validation data.

Do not present arbitrary weights as clinically validated.

Example workflow:

``` text
High Trust
    ↓
AI-assisted result

Medium Trust
    ↓
Mandatory human review

Low Trust
    ↓
Recapture / ophthalmologist
```

The important principle is **evidence-aware abstention**.

------------------------------------------------------------------------

# 11. Explicit AI Abstention

The system must be allowed to say:

> "I cannot reliably grade this image."

Possible reasons:

``` text
LOW IMAGE QUALITY
MODEL DISAGREEMENT
INSUFFICIENT LESION EVIDENCE
OUT-OF-DISTRIBUTION
LOW CALIBRATED CONFIDENCE
```

Never force the model to provide a confident grade when the evidence is
unreliable.

This should be treated as a core safety mechanism rather than an error
state.

------------------------------------------------------------------------

# 12. Out-of-Distribution Detection

A model can produce high softmax confidence on images that are unlike
its training distribution.

Implement an OOD mechanism using learned retinal embeddings.

Conceptual pipeline:

``` text
Fundus Image
     ↓
Retinal embedding
     ↓
Distance / density estimation
     ↓
OOD score
     ↓
Normal / suspicious
```

If the OOD score indicates an unfamiliar image:

``` text
High OOD
   ↓
Do not trust automated grading
   ↓
Human review / recapture
```

Evaluate OOD behavior using cross-dataset and artificially shifted
examples.

------------------------------------------------------------------------

# 13. Calibration and Uncertainty

The system's probability outputs should be calibrated.

Evaluate:

-   Expected Calibration Error (ECE),
-   Brier score,
-   reliability diagrams,
-   confidence vs actual correctness,
-   confidence vs evidence agreement.

Use temperature scaling or another validated calibration method where
appropriate.

The UI should distinguish:

``` text
Model probability
```

from:

``` text
Calibrated reliability
```

so that a probability such as 0.95 is not automatically treated as
clinically trustworthy.

------------------------------------------------------------------------

# 14. Structured Evidence Model

Extract structured retinal features from segmentation and analysis.

Candidate features:

``` text
lesion count
lesion area
lesion density
hemorrhage area
exudate area
microaneurysm density
vessel density
optic-disc relationship
macular-region involvement
lesion spatial distribution
texture/radiomic features
```

Candidate models:

``` text
TabPFN
CatBoost
LightGBM
```

Use this model as an independent structured-evidence branch.

A hybrid deep-visual + structured-evidence system can be evaluated
against an end-to-end visual model under domain shift.

------------------------------------------------------------------------

# 15. Evidence Fusion

The final prediction should combine:

``` text
                    ┌── Deep ordinal prediction
                    │
                    ├── Lesion evidence
                    │
Fundus ─────────────┼── Structured/radiomic features
                    │
                    ├── Image quality
                    │
                    ├── OOD score
                    │
                    └── Attribution/evidence agreement
                              │
                              ▼
                       Evidence Fusion
                              │
                              ▼
                       Trust/Risk Engine
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
              CLEAR         REVIEW        REFER
```

Do not simply average model outputs when the evidence conflicts.

Conflict should increase review priority.

------------------------------------------------------------------------

# 16. Offline-First Rural Deployment

The actual screening workflow should not depend on continuous Internet
access.

Target architecture:

``` text
Fundus Camera
     ↓
Android / Edge Device
     ↓
Local preprocessing
     ↓
Local AI inference
     ↓
Local report
     ↓
Encrypted local storage
     ↓
Sync when connectivity returns
```

Internet availability:

``` text
                INTERNET?
                  │
          ┌───────┴───────┐
          │               │
         YES              NO
          │               │
     Cloud sync        Local AI
          │               │
          └───────┬───────┘
                  ▼
           Patient record
```

The screening decision should remain available during temporary network
outages.

------------------------------------------------------------------------

# 17. Edge Inference

Prepare the inference stack for practical deployment.

Candidate stack:

``` text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime / optimized runtime
   ↓
Android / edge device
```

For MATLAB-oriented demonstration:

``` text
Trained model
   ↓
MATLAB-compatible inference
   ↓
Simulink integration
```

Measure:

-   inference latency,
-   memory usage,
-   model size,
-   CPU/GPU requirements,
-   energy/throughput where possible.

------------------------------------------------------------------------

# 18. Patient Workflow

Create an actual screening workflow rather than a model demo.

``` text
Patient ID
    ↓
Basic screening information
    ↓
Previous screening record
    ↓
Fundus capture
    ↓
Image quality assessment
    ↓
AI screening
    ↓
Risk classification
    ↓
Human review if required
    ↓
Referral decision
    ↓
Follow-up
```

Patient record:

``` text
Patient
 ├── previous screening
 ├── current screening
 ├── DR grade
 ├── lesion evidence
 ├── confidence
 ├── trust score
 ├── review status
 ├── referral status
 └── follow-up history
```

The system should preserve a clear audit trail.

------------------------------------------------------------------------

# 19. Longitudinal Monitoring

Use previous examinations when available.

Example:

``` text
2026 → Grade 1
2027 → Grade 2
2028 → Grade 3
```

The system should flag progression.

For paired images:

``` text
Current image
      +
Previous image
      ↓
Registration / comparison
      ↓
Change analysis
      ↓
Progression evidence
```

Do not claim that this predicts future disease unless a properly
validated predictive model is developed.

The initial objective is longitudinal change detection and follow-up
prioritization.

------------------------------------------------------------------------

# 20. District-Scale Simulation with MATLAB / Simulink

Simulink should demonstrate measurable healthcare impact.

Model:

``` text
Population
   ↓
Screening demand
   ↓
PHC acquisition rate
   ↓
Image quality failure rate
   ↓
AI throughput
   ↓
Referral rate
   ↓
Ophthalmologist capacity
   ↓
Review queue
   ↓
Waiting time
```

Compare operational scenarios:

### Scenario A --- Specialist-only

All cases are sent to ophthalmologists.

### Scenario B --- AI-only screening

AI processes all usable images.

### Scenario C --- AI + confidence-based triage

Only uncertain/high-risk cases are escalated.

### Scenario D --- AI + quality gate + evidence-based escalation

Quality control, evidence fusion and human review are all enabled.

Measure:

``` text
Patients screened
Cases referred
Specialist workload
Average waiting time
AI throughput
Processing latency
Bandwidth requirement
Human review volume
```

The key system-level question is:

> **How much specialist capacity can be redirected toward cases
> requiring expert attention while maintaining the required screening
> sensitivity?**

------------------------------------------------------------------------

# 21. Evaluation Strategy

Do not use accuracy as the primary success metric.

## Classification Metrics

-   Sensitivity
-   Specificity
-   ROC-AUC
-   PR-AUC
-   F1-score
-   Balanced accuracy
-   Quadratic Weighted Kappa (QWK)

## Screening Metrics

Most important:

-   sensitivity for referable DR,
-   specificity for referable DR.

The project should explicitly target the SIH26038 requirement of more
than 90% sensitivity and more than 85% specificity for referable DR.

## Calibration

-   ECE
-   Brier score
-   reliability diagram

## Segmentation

-   Dice
-   IoU
-   lesion-level sensitivity
-   lesion-level F1

## Robustness

Evaluate separately on:

``` text
clean images
blurred images
dark images
bright images
compressed images
cropped images
different camera characteristics
OOD images
cross-dataset test sets
```

## Deployment

Measure:

``` text
inference latency
model size
memory usage
offline capability
sync behavior
failure recovery
```

------------------------------------------------------------------------

# 22. Cross-Dataset Validation

Avoid relying only on random train/test splits from one dataset.

Preferred evaluation:

``` text
Training:
APTOS + EyePACS

External testing:
IDRiD
Messidor-2
```

and additional combinations where computationally feasible.

Also evaluate:

``` text
Training:
APTOS + IDRiD

External testing:
Messidor-2
```

The objective is to quantify domain shift.

A model that performs strongly in-domain but fails on another dataset
should not be described as robust.

------------------------------------------------------------------------

# 23. Ablation Study

Every major architectural component should be experimentally justified.

  System                  Referable Sensitivity   Specificity       QWK   Calibration
  --------------------- ----------------------- ------------- --------- -------------
  Baseline CNN                          Measure       Measure   Measure       Measure
  \+ Quality Gate                       Measure       Measure   Measure       Measure
  \+ Ordinal Head                       Measure       Measure   Measure       Measure
  \+ Lesion Evidence                    Measure       Measure   Measure       Measure
  \+ Evidence Fusion                    Measure       Measure   Measure       Measure
  \+ OOD / Abstention                   Measure       Measure   Measure       Measure

Do not fill this table with expected or fabricated values.

Run the experiments and populate it with measured results.

------------------------------------------------------------------------

# 24. Recommended Model Stack

## Image Quality

``` text
ResNet18 / EfficientNet-B0
```

## Main DR Model

Preferred:

``` text
RETFound
      ↓
CORAL ordinal head
```

Deployment-oriented alternative:

``` text
ConvNeXt-Tiny
      ↓
CORAL ordinal head
```

## Lesion Segmentation

``` text
SegFormer-B2 / B3
```

## Lesion Detection

If sufficient annotated data is available:

``` text
Deformable DETR
```

or an efficient object detector suitable for edge deployment.

## Structured Evidence

``` text
Radiomics
   +
TabPFN / CatBoost / LightGBM
```

The final model should be chosen after controlled experiments.

------------------------------------------------------------------------

# 25. Data Strategy

Use multiple public datasets for different purposes.

Potential datasets include:

-   APTOS 2019,
-   EyePACS,
-   IDRiD,
-   Messidor / Messidor-2,
-   other legally usable retinal datasets with appropriate labels.

Use datasets according to their available annotations.

For example:

``` text
DR grade datasets
        ↓
Ordinal classifier

Lesion pixel annotations
        ↓
Segmentation model

Lesion/object annotations
        ↓
Detection model

External datasets
        ↓
Robustness evaluation
```

Do not mix patient-level images between training and evaluation.

Avoid data leakage through duplicate or near-duplicate images.

Document:

-   dataset source,
-   license/usage restrictions,
-   number of images,
-   grade distribution,
-   annotation type,
-   train/validation/test split,
-   preprocessing,
-   augmentation.

------------------------------------------------------------------------

# 26. Handling Class Imbalance

DR datasets are usually imbalanced.

Use methods such as:

-   class-balanced sampling,
-   weighted losses,
-   focal/ordinal loss where appropriate,
-   augmentation,
-   threshold tuning,
-   stratified evaluation.

Do not optimize only for overall accuracy because majority-class
performance can hide clinically important failures.

------------------------------------------------------------------------

# 27. Robustness Testing

Create a dedicated robustness benchmark.

``` text
                 ORIGINAL IMAGE
                       │
      ┌────────────────┼────────────────┐
      │                │                │
    Blur          Illumination      Compression
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                 Model response
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Stable result       Uncertain
                                │
                                ▼
                           Human review
```

For every corruption type measure:

-   grade stability,
-   confidence change,
-   trust-score change,
-   OOD behavior,
-   referral sensitivity.

The system should become more conservative when image reliability
deteriorates.

------------------------------------------------------------------------

# 28. Safety-Oriented Decision Logic

A possible high-level decision engine:

``` text
IF image_quality < minimum
    → RECATURE

ELSE IF OOD score is high
    → HUMAN REVIEW

ELSE IF evidence disagreement is high
    → HUMAN REVIEW

ELSE IF calibrated confidence is low
    → HUMAN REVIEW

ELSE IF referable DR probability exceeds validated threshold
    → REFERRAL WORKFLOW

ELSE
    → AI-ASSISTED SCREENING RESULT
```

Thresholds must be determined experimentally on validation data.

The decision engine should be transparent and auditable.

------------------------------------------------------------------------

# 29. Dashboard / Product Interface

Build a professional clinical-style interface.

## Screen 1 --- Patient

``` text
Patient ID
Screening date
Previous screening
Capture status
```

## Screen 2 --- Image Quality

``` text
Focus
Illumination
Coverage
Artifacts
Overall quality
```

## Screen 3 --- AI Analysis

Show side-by-side:

``` text
Original image
Lesion segmentation
Grad-CAM / attribution
Lesion overlay
```

## Screen 4 --- DR Result

``` text
Grade: 2 — Moderate NPDR

P(≥1): 98%
P(≥2): 91%
P(≥3): 18%
P(≥4):  3%
```

## Screen 5 --- Evidence

``` text
Microaneurysms: 17
Exudates: 5
Hemorrhages: 2

Evidence agreement: 88%
OOD status: LOW
```

## Screen 6 --- Trust

``` text
TRUST SCORE
87 / 100

ACTION
REFER TO OPHTHALMOLOGIST
```

## Screen 7 --- Longitudinal

``` text
Previous grade
Current grade
Change
Follow-up status
```

## Screen 8 --- District Analytics

``` text
Patients screened
Referral rate
Human review load
Average latency
Estimated specialist workload
```

------------------------------------------------------------------------

# 30. Backend Architecture

Recommended structure:

``` text
                    Frontend
                       │
                       ▼
                    FastAPI
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     Inference      Patient DB      Sync API
        │              │              │
        ▼              ▼              ▼
     AI Engine       PostgreSQL     Cloud Storage
        │
 ┌──────┼────────┬──────────┬─────────┐
 ▼      ▼        ▼          ▼         ▼
Quality DR      Lesion   Structured  OOD/
Model   Model   Model     Evidence   Trust
```

Use modular services so models can be replaced independently.

------------------------------------------------------------------------

# 31. Privacy and Security

The system handles sensitive health information.

Implement:

-   role-based access control,
-   encrypted storage,
-   encrypted transport,
-   audit logs,
-   minimum necessary patient data,
-   local encryption for offline records,
-   secure synchronization,
-   session management,
-   data retention policies,
-   separation of identifiers and clinical data where appropriate.

Possible roles:

``` text
PHC Operator
    ↓
Capture + view screening status

Ophthalmologist
    ↓
Review AI evidence + finalize assessment

District Administrator
    ↓
View aggregated analytics

System Administrator
    ↓
System management + audit
```

------------------------------------------------------------------------

# 32. Human-in-the-Loop Review Queue

The system should generate a prioritized queue.

Priority can depend on:

``` text
High referable probability
+
Low trust
+
Evidence disagreement
+
High OOD score
+
Image quality concerns
+
Longitudinal progression
```

Example:

``` text
REVIEW QUEUE

1. Patient A
   High-risk + evidence disagreement

2. Patient B
   High OOD score

3. Patient C
   High referable probability

4. Patient D
   Moderate confidence
```

The queue should help ophthalmologists focus on the cases requiring
attention.

------------------------------------------------------------------------

# 33. What Not to Do

Do not build the project around:

-   a collection of many unrelated classifiers,
-   Grad-CAM alone as the innovation,
-   a single accuracy number,
-   a single-dataset random split,
-   arbitrary confidence thresholds,
-   fabricated clinical validation,
-   segmentation presented as diagnosis,
-   cloud-only inference,
-   a decorative Simulink block diagram.

The project should demonstrate engineering depth through measurable
reliability and operational performance.

------------------------------------------------------------------------

# 34. Core Innovation Stack

The project should aim to implement these pillars:

``` text
1. QUALITY-AWARE
   ↓
   Reject or improve unreliable images.

2. ORDINAL
   ↓
   Model DR severity as an ordered progression.

3. MULTI-EVIDENCE
   ↓
   Combine visual, lesion and structured evidence.

4. CALIBRATED
   ↓
   Make confidence measurable.

5. OOD-AWARE
   ↓
   Recognize unfamiliar inputs.

6. ABSTENTION
   ↓
   Allow AI to request human review.

7. OFFLINE-FIRST
   ↓
   Work without continuous connectivity.

8. LONGITUDINAL
   ↓
   Track changes across screenings.

9. DISTRICT SIMULATION
   ↓
   Quantify operational healthcare impact.
```

------------------------------------------------------------------------

# 35. Development Priority

Do not build every feature at the same time.

## Phase 1 --- Core AI

``` text
Fundus image
    ↓
Quality model
    ↓
Ordinal DR model
    ↓
Basic report
```

## Phase 2 --- Evidence

``` text
SegFormer
    +
Lesion features
    +
Evidence fusion
```

## Phase 3 --- Reliability

``` text
Calibration
    +
OOD detection
    +
Trust score
    +
Abstention
```

## Phase 4 --- Product

``` text
FastAPI
    +
Frontend
    +
Patient records
    +
Review queue
```

## Phase 5 --- Deployment

``` text
ONNX
    +
Offline inference
    +
Sync
```

## Phase 6 --- Systems Engineering

``` text
Longitudinal tracking
    +
Simulink district simulation
```

## Phase 7 --- Validation

``` text
Cross-dataset testing
    +
Robustness testing
    +
Ablation study
    +
Calibration analysis
```

------------------------------------------------------------------------

# 36. Final Project Positioning

The project should be presented as:

> **An evidence-aware, reliability-aware and offline-first AI screening
> platform for diabetic retinopathy in rural healthcare settings.**

The system is not simply:

> "AI that detects diabetic retinopathy."

It is:

> **A screening system that evaluates image quality, estimates DR
> severity using ordinal reasoning, identifies lesion-level evidence,
> cross-checks independent evidence sources, calibrates uncertainty,
> detects unfamiliar inputs, abstains when reliability is insufficient,
> operates offline, supports ophthalmologist review, tracks longitudinal
> change, and models district-level healthcare capacity.**

The strongest technical story is:

> **The system knows when its evidence is trustworthy, explains that
> evidence at lesion level, refuses unreliable predictions, works under
> rural imaging constraints, and intelligently allocates scarce
> ophthalmologist attention.**

------------------------------------------------------------------------

# 37. Research Basis

The architecture should be grounded in current medical-AI research
rather than being presented as a collection of arbitrary technologies.

## Retinal foundation models

RETFound and subsequent research support large-scale retinal pretraining
and transfer to ophthalmic tasks.

## Ordinal DR grading

Recent DR research has explored CORAL-style ordinal formulations because
DR severity has an inherent ordering and cross-dataset reliability is
important.

## Fundus image quality

Retinal image-quality research supports automated quality assessment and
emphasizes that quality criteria vary across datasets and acquisition
conditions.

## Lesion segmentation and detection

Recent multi-task retinal research has investigated transformer-based
segmentation/detection architectures including SegFormer and Deformable
DETR.

## Structured evidence

Recent research has demonstrated that structured/radiomic retinal
features can complement end-to-end visual models, especially when
external validation is considered.

## Human-assisted screening

Indian DR screening guidance emphasizes AI as an assistive component of
screening workflows, with clinical review and appropriate referral
pathways.

------------------------------------------------------------------------

# 38. Research References

1.  **A data-efficient strategy for building high-performing medical
    foundation models --- Nature Biomedical Engineering**
    https://www.nature.com/articles/s41551-025-01365-0

2.  **Toward Reliable Diabetic Retinopathy Screening --- PubMed**
    https://pubmed.ncbi.nlm.nih.gov/42515403/

3.  **Multi-task transformer framework and radiomic signatures for
    multi-lesion segmentation, detection, and grading in diabetic
    retinopathy --- ScienceDirect**
    https://www.sciencedirect.com/science/article/pii/S1572100026001225

4.  **Image quality assessment of retinal fundus photographs for
    diabetic retinopathy in the machine learning era --- Eye / Nature**
    https://www.nature.com/articles/s41433-023-02717-3

5.  **FundaQ-8: A Clinically-Inspired Scoring Framework for Automated
    Fundus Image Quality Assessment --- arXiv**
    https://arxiv.org/abs/2506.20303

6.  **Diabetic retinopathy screening guidelines for Physicians in India
    --- RSSDI/VRSI Position Statement**
    https://link.springer.com/article/10.1007/s13410-023-01296-z

7.  **Artificial intelligence for advancing eye care in resource-poor
    settings: Assessing predictive accuracy of an AI model for diabetic
    retinopathy screening in India --- PMC**
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12177155/

8.  **Development and Validation of a Deep Learning Algorithm for
    Detection of Diabetic Retinopathy in Retinal Fundus Photographs ---
    JAMA** https://jamanetwork.com/journals/jama/fullarticle/2588763

------------------------------------------------------------------------

# 39. Final Build Target

The final working system should look like:

``` text
                    ┌─────────────────────┐
                    │    FUNDUS IMAGE     │
                    └──────────┬──────────┘
                               ↓
                       ┌───────────────┐
                       │  QUALITY AI   │
                       └───────┬───────┘
                               ↓
                     QUALITY ACCEPTED?
                       /             \
                     NO               YES
                     ↓                 ↓
                 RECAPTURE       STANDARDIZATION
                                       ↓
                         ┌─────────────┴─────────────┐
                         ↓                           ↓
                  ORDINAL DR MODEL            LESION ANALYSIS
                         ↓                           ↓
                  Grade 0–4                  MA / EX / HE
                         │                           │
                         └─────────────┬─────────────┘
                                       ↓
                                STRUCTURED FEATURES
                                       ↓
                                EVIDENCE FUSION
                                       ↓
                              CALIBRATION + OOD
                                       ↓
                                  TRUST SCORE
                                       ↓
                         ┌─────────────┼─────────────┐
                         ↓             ↓             ↓
                       CLEAR         REVIEW        HIGH RISK
                         ↓             ↓             ↓
                    AI RESULT     OPHTHALMOLOGIST   REFERRAL
                                       ↓
                                 FINAL DECISION
                                       ↓
                              PATIENT RECORD
                                       ↓
                              LONGITUDINAL DATA
                                       ↓
                                DISTRICT MODEL
                                       ↓
                                  SIMULINK
```

## Definition of Done

The project is considered technically complete when it can demonstrate:

-   reliable fundus image quality assessment,
-   ordinal DR grading,
-   lesion-level evidence,
-   structured evidence fusion,
-   calibrated probabilities,
-   OOD detection,
-   explicit AI abstention,
-   human-review escalation,
-   offline inference,
-   patient-level workflow,
-   longitudinal comparison,
-   cross-dataset evaluation,
-   corruption/robustness testing,
-   ablation study,
-   latency and deployment measurements,
-   and a Simulink model demonstrating district-level operational
    impact.

The implementation should prioritize **measured reliability,
reproducibility and deployability** over adding more model names.
