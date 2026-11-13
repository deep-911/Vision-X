%% =========================================================================
%% VisionX: Two-Stage Model Evaluator (Calibrated Clinical Triage)
%% Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
%% =========================================================================
% Evaluates the Two-Stage Model on the 733-scan holdout set:
%   - Stage 1: Referral Gate (ResNet-18) with calibrated triage threshold
%   - Stage 2: Severity Stagers (Stager A: No DR vs Mild, Stager B: Moderate vs Severe vs PDR)
% Saves: reports/two_stage_evaluation.json
% =========================================================================

function results = evaluate_two_stage()
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    fprintf('===================================================================\n');
    fprintf(' VisionX: Two-Stage Model Evaluator (Calibrated Clinical Triage)\n');
    fprintf(' Acceleration: NVIDIA GeForce RTX 3050 Laptop GPU\n');
    fprintf('===================================================================\n\n');

    gatePath = fullfile(projectRoot, 'models', 'production', 'visionx_stage1_gate.mat');
    if ~exist(gatePath, 'file')
        gatePath = fullfile(projectRoot, 'models', 'visionx_stage1_gate.mat');
    end

    stagerPath = fullfile(projectRoot, 'models', 'production', 'visionx_stage2_stager.mat');
    if ~exist(stagerPath, 'file')
        stagerPath = fullfile(projectRoot, 'models', 'visionx_stage2_stager.mat');
    end

    dataFolder = fullfile(projectRoot, 'data', 'processed', 'aptos_true_512');
    if ~exist(dataFolder, 'dir')
        dataFolder = fullfile(projectRoot, 'data', 'aptos_true_512');
    end

    if ~exist(gatePath, 'file')
        error('Referral Gate model not found at %s', gatePath);
    end
    if ~exist(stagerPath, 'file')
        error('Specialist Stager model not found at %s', stagerPath);
    end

    %% 1. LOAD MODELS
    fprintf('[1/5] Loading Pipeline Models into Memory...\n');
    gateData = load(gatePath);
    netGate = gateData.trainedNet;
    fprintf('  - Stage 1 Referral Gate loaded.\n');

    stagerData = load(stagerPath);
    netStagerA = stagerData.trainedStagerA;
    netStagerB = stagerData.trainedStagerB;
    fprintf('  - Stage 2 Severity Stagers A & B loaded.\n');

    %% 2. LOAD 733-SCAN HOLDOUT COHORT
    fprintf('\n[2/5] Loading 733-Scan Canonical Validation Cohort (Seed 42)...\n');
    clinicalOrder = {'No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'};
    imdsRaw = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    imdsRaw.Labels = reordercats(imdsRaw.Labels, clinicalOrder);

    rng(42, 'twister');
    [~, imdsVal] = splitEachLabel(imdsRaw, 0.8, 'randomized');
    numScans = numel(imdsVal.Files);
    fprintf('  - %d validation scans ready for inference.\n', numScans);

    inputSize = [224, 224];
    augVal = augmentedImageDatastore(inputSize, imdsVal);

    %% 3. FORWARD INFERENCE & PROBABILITY EXTRACTION
    fprintf('\n[3/5] Running Forward Inference across All Networks on GPU...\n');
    
    % Stage 1 Gate probabilities: [P(Non_Referable), P(Referable)]
    fprintf('  - Step 1: Evaluating Stage 1 Referral Gate posterior probabilities...\n');
    [~, gateScores] = classify(netGate, augVal);
    pRef = gateScores(:, 2); % P(Referable)

    % Stage 2A Stager probabilities: [P(No_DR), P(Mild)]
    fprintf('  - Step 2: Evaluating Stage 2A Non-Referable Stager...\n');
    [~, scoresA] = classify(netStagerA, augVal);

    % Stage 2B Stager probabilities: [P(Moderate), P(Severe), P(PDR)]
    fprintf('  - Step 3: Evaluating Stage 2B Referable Stager...\n');
    [~, scoresB] = classify(netStagerB, augVal);

    %% 4. STAGE 1 THRESHOLD SENSITIVITY SWEEP (OPERATING CURVE)
    fprintf('\n[4/5] Operating Point Sweep on Stage 1 Referral Gate...\n');
    YTrue = imdsVal.Labels;
    referableTrue = double(YTrue ~= 'No_DR' & YTrue ~= 'Mild');
    totalSick = sum(referableTrue == 1);
    totalHealthy = sum(referableTrue == 0);

    fprintf('  Total Sick Patients (Grade >= 2): %d\n', totalSick);
    fprintf('  Total Healthy/Mild (Grade < 2)  : %d\n\n', totalHealthy);

    fprintf('  %-10s | %-12s | %-12s | %-12s | %-12s\n', ...
        'Threshold', 'Sensitivity', 'Specificity', 'Caught/Sick', 'Missed(FN)');
    fprintf('  -----------------------------------------------------------------\n');

    testThresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20];
    for t = testThresholds
        predsRef_t = double(pRef >= t);
        tp_t = sum(predsRef_t == 1 & referableTrue == 1);
        fn_t = sum(predsRef_t == 0 & referableTrue == 1);
        tn_t = sum(predsRef_t == 0 & referableTrue == 0);
        fp_t = sum(predsRef_t == 1 & referableTrue == 0);
        sens_t = tp_t / (tp_t + fn_t);
        spec_t = tn_t / (tn_t + fp_t);
        fprintf('  tau = %-5.2f | %-10.2f%% | %-10.2f%% | %-12s | %-12d\n', ...
            t, sens_t * 100, spec_t * 100, sprintf('%d/%d', tp_t, totalSick), fn_t);
    end

    %% 5. TWO-STAGE MODEL DECISION WITH CALIBRATED TRIAGE
    fprintf('\n[5/5] Generating Final Diagnoses with Calibrated Triage Threshold...\n');
    tau_referable = 0.30; % Calibrated threshold (protects against false negatives)
    finalPreds = cell(numScans, 1);
    
    for i = 1:numScans
        if pRef(i) < tau_referable
            % Screened as Non-Referable -> Stager A decides between Grade 0 and Grade 1
            if scoresA(i, 1) >= scoresA(i, 2)
                finalPreds{i} = 'No_DR';
            else
                finalPreds{i} = 'Mild';
            end
        else
            % Screened as Referable -> Stager B decides between Grades 2, 3, 4
            [~, bestBIdx] = max(scoresB(i, :));
            refSubClasses = {'Moderate', 'Severe', 'Proliferate_DR'};
            finalPreds{i} = refSubClasses{bestBIdx};
        end
    end

    finalPredsCategorical = reordercats(categorical(finalPreds), clinicalOrder);

    %% 6. FINAL BENCHMARK EVALUATION
    confMat = confusionmat(YTrue, finalPredsCategorical, 'Order', clinicalOrder);
    exactAccuracy = sum(diag(confMat)) / sum(confMat(:));
    qwk = compute_qwk(confMat);

    referablePred = double(finalPredsCategorical ~= 'No_DR' & finalPredsCategorical ~= 'Mild');
    tp = sum(referablePred == 1 & referableTrue == 1);
    fn = sum(referablePred == 0 & referableTrue == 1);
    tn = sum(referablePred == 0 & referableTrue == 0);
    fp = sum(referablePred == 1 & referableTrue == 0);
    sens = tp / max(tp + fn, 1);
    spec = tn / max(tn + fp, 1);

    fprintf('\n===================================================================\n');
    fprintf(' TWO-STAGE MODEL BENCHMARK RESULTS\n');
    fprintf('===================================================================\n');
    fprintf('  * Total Validation Scans Tested : %d\n', numScans);
    fprintf('  * Multi-Class Exact Accuracy    : %.2f%%\n', exactAccuracy * 100);
    fprintf('  * Quadratic Weighted Kappa (QWK): %.4f\n', qwk);
    fprintf('  * Referable DR Sensitivity      : %.2f%% (Target >= 90%%)\n', sens * 100);
    fprintf('  * Referable DR Specificity      : %.2f%% (Target >= 85%%)\n', spec * 100);
    fprintf('  * Caught Sick Patients          : %d / %d\n', tp, tp + fn);
    fprintf('  * False Negatives (Missed)      : %d patients\n', fn);
    fprintf('===================================================================\n\n');

    fprintf('Two-Stage Model 5x5 Confusion Matrix:\n');
    disp(array2table(confMat, 'RowNames', clinicalOrder, 'VariableNames', clinicalOrder));

    recalls = diag(confMat) ./ sum(confMat, 2);
    fprintf('\nPer-Grade Sensitivity (Recall):\n');
    for c = 1:5
        fprintf('  - %-16s: %.2f%%\n', clinicalOrder{c}, recalls(c) * 100);
    end

    %% SAVE REPORT
    reportDir = fullfile(projectRoot, 'artifacts', 'reports');
    if ~exist(reportDir, 'dir'); mkdir(reportDir); end
    reportPath = fullfile(reportDir, 'two_stage_evaluation.json');

    metrics = struct();
    metrics.model_name = 'Two-Stage Model (Stage 1 Gate + Stage 2 Stagers)';
    metrics.tau_referable = tau_referable;
    metrics.accuracy = exactAccuracy;
    metrics.qwk = qwk;
    metrics.referable_sensitivity = sens;
    metrics.referable_specificity = spec;
    metrics.caught_sick = tp;
    metrics.total_sick = tp + fn;
    metrics.false_negatives = fn;
    metrics.confusion_matrix = confMat;
    metrics.classes = clinicalOrder;

    jsonStr = jsonencode(metrics, 'PrettyPrint', true);
    fid = fopen(reportPath, 'w');
    if fid ~= -1
        fwrite(fid, jsonStr, 'char');
        fclose(fid);
        fprintf('\n[SAVE] Evaluation report saved to: %s\n', reportPath);
    end
    
    % Also save to apex_evaluation.json for backward compatibility
    fid2 = fopen(fullfile(reportDir, 'apex_evaluation.json'), 'w');
    if fid2 ~= -1
        fwrite(fid2, jsonStr, 'char');
        fclose(fid2);
    end

    results = metrics;
end

function qwk = compute_qwk(cm)
    K = size(cm, 1);
    [I, J] = meshgrid(1:K, 1:K);
    W = ((I - J).^2) / ((K - 1)^2);
    rowTotals = sum(cm, 2);
    colTotals = sum(cm, 1);
    E = (rowTotals * colTotals) / sum(cm(:));
    qwk = 1 - (sum(sum(W .* cm)) / (sum(sum(W .* E)) + 1e-12));
end
