% MATLAB Smoke Test script to verify PyTorch -> ONNX compatibility
disp('Starting MATLAB ONNX Smoke Test...');

% Path to the exported ONNX model
model_path = 'artifacts/exports/baseline_softmax.onnx';

try
    % Load the network
    net = importONNXNetwork(model_path, 'OutputLayerType', 'regression');
    disp('Successfully imported ONNX model into MATLAB.');
    
    % Create a dummy input matching the expected dimensions
    % PyTorch is NCHW. We create a 224x224x3 array of ones.
    dummy_image = ones(224, 224, 3, 'single');
    dl_image = dlarray(dummy_image, 'SSC'); % Spatial, Spatial, Channel
    
    % Run prediction
    preds = predict(net, dl_image);
    
    % Since output is just logits, apply softmax
    preds_exp = exp(preds);
    probs = preds_exp ./ sum(preds_exp);
    
    disp('MATLAB Predictions (Probabilities):');
    disp(extractdata(probs));
    
    disp('Smoke Test PASSED.');
catch ME
    disp('Smoke Test FAILED with error:');
    disp(ME.message);
    exit(1);
end
exit(0);
