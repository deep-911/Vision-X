%% =========================================================================
%% VisionX: Stage 2 Specialist Severity Stagers (Referable & Non-Referable)
%% Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
%% =========================================================================
% Trains two decoupled specialist sub-networks on the true 512x512 dataset:
%   - Stager A (Non-Referable): Differentiates No_DR (Grade 0) vs Mild (Grade 1)
%   - Stager B (Referable)    : Differentiates Moderate (Grade 2) vs Severe (Grade 3) vs PDR (Grade 4)
% Saves: models/visionx_stage2_stager.mat & reports/stage2_stager_evaluation.json
% =========================================================================

function results = train_stage2_stager()
    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    fprintf('===================================================================\n');
    fprintf(' VisionX: Stage 2 Specialist Severity Stagers (True 512x512)\n');
    fprintf('===================================================================\n\n');

    dataFolder = fullfile(projectRoot, 'data', 'processed', 'aptos_true_512');
    if ~exist(dataFolder, 'dir')
        dataFolder = fullfile(projectRoot, 'data', 'aptos_true_512');
    end

    %% ---------------------------------------------------------------------
    %% PART A: TRAIN REFERABLE SEVERITY STAGER (MODERATE vs SEVERE vs PDR)
    %% ---------------------------------------------------------------------
    fprintf('[PART A] Configuring Referable Specialist Stager (Grades 2, 3, 4)...\n');
    referableClasses = {'Moderate', 'Severe', 'Proliferate_DR'};
    
    imdsRefAll = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    keepIdx = ismember(cellstr(imdsRefAll.Labels), referableClasses);
    imdsRef = imageDatastore(imdsRefAll.Files(keepIdx));
    imdsRef.Labels = reordercats(categorical(cellstr(imdsRefAll.Labels(keepIdx))), referableClasses);

    fprintf('  Total Referable Scans: %d\n', numel(imdsRef.Files));
    disp(countEachLabel(imdsRef));

    rng(42, 'twister');
    [trainRef, valRef] = splitEachLabel(imdsRef, 0.8, 'randomized');

    % Balance Severe (+2x) and PDR (+1x)
    idxSev = find(trainRef.Labels == 'Severe');
    idxPdr = find(trainRef.Labels == 'Proliferate_DR');
    extraRefFiles = [repmat(trainRef.Files(idxSev), 2, 1); repmat(trainRef.Files(idxPdr), 1, 1)];
    extraRefLabels = [repmat(trainRef.Labels(idxSev), 2, 1); repmat(trainRef.Labels(idxPdr), 1, 1)];

    allTrainRefFiles = [trainRef.Files; extraRefFiles];
    allTrainRefLabels = [trainRef.Labels; extraRefLabels];
    trainRef = imageDatastore(allTrainRefFiles);
    trainRef.Labels = reordercats(categorical(allTrainRefLabels), referableClasses);

    fprintf('  Balanced Referable Training Set: %d scans\n', numel(trainRef.Files));
    disp(countEachLabel(trainRef));

    netB = resnet18;
    inputSize = [224, 224];
    lgraphB = layerGraph(netB);
    newFcB = fullyConnectedLayer(3, 'Name', 'visionx_stagerB_fc', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    newSoftmaxB = softmaxLayer('Name', 'visionx_stagerB_softmax');
    newOutputB = classificationLayer('Name', 'visionx_stagerB_output');

    lgraphB = replaceLayer(lgraphB, 'fc1000', newFcB);
    lgraphB = replaceLayer(lgraphB, 'fc1000_softmax', newSoftmaxB);
    lgraphB = replaceLayer(lgraphB, 'ClassificationLayer_fc1000', newOutputB);

    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandYReflection', true, ...
        'RandRotation', [-20, 20], ...
        'RandScale', [0.88, 1.12]);

    augTrainRef = augmentedImageDatastore(inputSize, trainRef, 'DataAugmentation', imageAugmenter);
    augValRef = augmentedImageDatastore(inputSize, valRef);

    optsB = trainingOptions('adam', ...
        'ExecutionEnvironment', 'gpu', ...
        'InitialLearnRate', 1e-4, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 2, ...
        'LearnRateDropFactor', 0.3, ...
        'MaxEpochs', 4, ...
        'MiniBatchSize', 24, ...
        'ValidationData', augValRef, ...
        'ValidationFrequency', 25, ...
        'Verbose', true, ...
        'Plots', 'none');

    fprintf('  Training Stager B on GPU...\n');
    trainedStagerB = trainNetwork(augTrainRef, lgraphB, optsB);

    [predRef, scoresRef] = classify(trainedStagerB, augValRef);
    cmRef = confusionmat(valRef.Labels, predRef, 'Order', referableClasses);
    accRef = sum(diag(cmRef)) / sum(cmRef(:));
    fprintf('  --> Referable Stager Exact Accuracy: %.2f%%\n', accRef * 100);
    disp(array2table(cmRef, 'RowNames', referableClasses, 'VariableNames', referableClasses));

    %% ---------------------------------------------------------------------
    %% PART B: TRAIN NON-REFERABLE STAGER (NO_DR vs MILD)
    %% ---------------------------------------------------------------------
    fprintf('\n[PART B] Configuring Non-Referable Specialist Stager (No_DR vs Mild)...\n');
    nonRefClasses = {'No_DR', 'Mild'};

    imdsNonRefAll = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    keepIdxNon = ismember(cellstr(imdsNonRefAll.Labels), nonRefClasses);
    imdsNonRef = imageDatastore(imdsNonRefAll.Files(keepIdxNon));
    imdsNonRef.Labels = reordercats(categorical(cellstr(imdsNonRefAll.Labels(keepIdxNon))), nonRefClasses);

    [trainNon, valNon] = splitEachLabel(imdsNonRef, 0.8, 'randomized');

    % Balance Mild (+3x)
    idxMild = find(trainNon.Labels == 'Mild');
    extraNonFiles = repmat(trainNon.Files(idxMild), 3, 1);
    extraNonLabels = repmat(trainNon.Labels(idxMild), 3, 1);

    allTrainNonFiles = [trainNon.Files; extraNonFiles];
    allTrainNonLabels = [trainNon.Labels; extraNonLabels];
    trainNon = imageDatastore(allTrainNonFiles);
    trainNon.Labels = reordercats(categorical(allTrainNonLabels), nonRefClasses);

    netA = resnet18;
    lgraphA = layerGraph(netA);
    newFcA = fullyConnectedLayer(2, 'Name', 'visionx_stagerA_fc', ...
        'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    newSoftmaxA = softmaxLayer('Name', 'visionx_stagerA_softmax');
    newOutputA = classificationLayer('Name', 'visionx_stagerA_output');

    lgraphA = replaceLayer(lgraphA, 'fc1000', newFcA);
    lgraphA = replaceLayer(lgraphA, 'fc1000_softmax', newSoftmaxA);
    lgraphA = replaceLayer(lgraphA, 'ClassificationLayer_fc1000', newOutputA);

    augTrainNon = augmentedImageDatastore(inputSize, trainNon, 'DataAugmentation', imageAugmenter);
    augValNon = augmentedImageDatastore(inputSize, valNon);

    optsA = trainingOptions('adam', ...
        'ExecutionEnvironment', 'gpu', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 3, ...
        'MiniBatchSize', 32, ...
        'ValidationData', augValNon, ...
        'ValidationFrequency', 25, ...
        'Verbose', true, ...
        'Plots', 'none');

    fprintf('  Training Stager A on GPU...\n');
    trainedStagerA = trainNetwork(augTrainNon, lgraphA, optsA);

    [predNon, scoresNon] = classify(trainedStagerA, augValNon);
    cmNon = confusionmat(valNon.Labels, predNon, 'Order', nonRefClasses);
    accNon = sum(diag(cmNon)) / sum(cmNon(:));
    fprintf('  --> Non-Referable Stager Exact Accuracy: %.2f%%\n', accNon * 100);
    disp(array2table(cmNon, 'RowNames', nonRefClasses, 'VariableNames', nonRefClasses));

    %% SAVE BOTH SPECIALIST STAGERS & REPORT
    modelDir = fullfile(projectRoot, 'models', 'production');
    if ~exist(modelDir, 'dir'); mkdir(modelDir); end
    modelPath = fullfile(modelDir, 'visionx_stage2_stager.mat');
    save(modelPath, 'trainedStagerA', 'trainedStagerB');
    fprintf('\n[SAVE] Specialist Stagers saved to: %s\n', modelPath);

    reportDir = fullfile(projectRoot, 'artifacts', 'reports');
    if ~exist(reportDir, 'dir'); mkdir(reportDir); end
    reportPath = fullfile(reportDir, 'stage2_stager_evaluation.json');

    rep = struct();
    rep.model_type = 'Stage 2 Specialist Severity Stagers (ResNet-18)';
    rep.accRef = accRef;
    rep.accNon = accNon;
    rep.classes_referable = referableClasses;
    rep.classes_non_referable = nonRefClasses;
    jsonStr = jsonencode(rep, 'PrettyPrint', true);
    fid = fopen(reportPath, 'w');
    if fid ~= -1
        fwrite(fid, jsonStr, 'char');
        fclose(fid);
        fprintf('[SAVE] Specialist Stager report saved to: %s\n', reportPath);
    end

    results = struct('accRef', accRef, 'accNon', accNon);
end
