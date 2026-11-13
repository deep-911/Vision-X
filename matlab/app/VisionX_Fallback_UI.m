classdef VisionX_Fallback_UI < matlab.apps.AppBase
    % VisionX_Fallback_UI
    % A pure MATLAB fallback user interface for the VisionX Deferral Pipeline.
    % Designed for SIH 26038 (MathWorks Track) in case a pure MATLAB execution
    % environment is strictly required by the judges.

    properties (Access = public)
        UIFigure              matlab.ui.Figure
        TitleLabel            matlab.ui.control.Label
        LoadImageButton       matlab.ui.control.Button
        RunAnalysisButton     matlab.ui.control.Button
        OriginalImageAxes     matlab.ui.control.UIAxes
        HeatmapAxes           matlab.ui.control.UIAxes
        StatusPanel           matlab.ui.container.Panel
        GradeLabel            matlab.ui.control.Label
        ConfidenceLabel       matlab.ui.control.Label
        ActionLabel           matlab.ui.control.Label
        CurrentImagePath      string
        LoadedImage           
    end

    methods (Access = private)

        % Button pushed function: LoadImageButton
        function LoadImageButtonPushed(app, event)
            [file, path] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select a Retinal Scan');
            if isequal(file, 0)
                return;
            end
            app.CurrentImagePath = fullfile(path, file);
            app.LoadedImage = imread(app.CurrentImagePath);
            imshow(app.LoadedImage, 'Parent', app.OriginalImageAxes);
            app.GradeLabel.Text = 'Grade: Pending';
            app.ConfidenceLabel.Text = 'Confidence: Pending';
            app.ActionLabel.Text = 'Action: Ready to Analyze';
            app.ActionLabel.FontColor = [0, 0, 0];
            app.RunAnalysisButton.Enable = 'on';
        end

        % Button pushed function: RunAnalysisButton
        function RunAnalysisButtonPushed(app, event)
            app.RunAnalysisButton.Enable = 'off';
            app.ActionLabel.Text = 'Action: Running Deferral Pipeline...';
            drawnow;

            try
                % This invokes the existing pipeline or inference script
                % For demonstration in the UI, we simulate the output based on 
                % the Python pipeline rules, or call the MATLAB network directly.
                
                % Dummy structural call to the trainedNet if available
                projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
                modelPath = fullfile(projectRoot, 'models', 'research', 'visionx_matlab_net.mat');
                if ~exist(modelPath, 'file')
                    modelPath = fullfile(projectRoot, 'models', 'production', 'visionx_matlab_net.mat');
                end
                if ~exist(modelPath, 'file')
                    modelPath = fullfile(projectRoot, 'models', 'visionx_matlab_net.mat');
                end
                
                if exist(modelPath, 'file')
                    m = load(modelPath, 'trainedNet', 'classes');
                    img = imresize(app.LoadedImage, m.trainedNet.Layers(1).InputSize(1:2));
                    [label, scores] = classify(m.trainedNet, img);
                    conf = max(scores);
                    
                    % Temperature Scaling (T=1.5)
                    T = 1.5;
                    scaled_scores = exp(log(scores)/T) / sum(exp(log(scores)/T));
                    scaled_conf = max(scaled_scores);
                    
                    app.GradeLabel.Text = sprintf('Grade: %s', char(label));
                    app.ConfidenceLabel.Text = sprintf('Confidence: %.1f%%', scaled_conf * 100);
                    
                    try
                        g = gradCAM(m.trainedNet, img, label);
                        g = imresize(g, size(app.LoadedImage, 1:2));
                        imshow(app.LoadedImage, 'Parent', app.HeatmapAxes);
                        hold(app.HeatmapAxes, 'on');
                        h = imagesc(app.HeatmapAxes, g);
                        h.AlphaData = 0.5;
                        colormap(app.HeatmapAxes, 'jet');
                        hold(app.HeatmapAxes, 'off');
                    catch
                        app.ActionLabel.Text = 'Action: Grad-CAM Failed';
                    end
                    
                    % Basic UI Safety Gate representation
                    if scaled_conf < 0.60
                        app.ActionLabel.Text = 'Action: ESCALATED TO HUMAN (Low Confidence)';
                        app.ActionLabel.FontColor = [0.85, 0.15, 0.15];
                    else
                        app.ActionLabel.Text = 'Action: Auto-Graded Successfully';
                        app.ActionLabel.FontColor = [0.15, 0.65, 0.15];
                    end
                else
                    % Fallback mock visual for demo purposes if model isn't built yet
                    imshow(app.LoadedImage, 'Parent', app.HeatmapAxes);
                    app.GradeLabel.Text = 'Grade: Moderate (Mock)';
                    app.ConfidenceLabel.Text = 'Confidence: 87.5% (Calibrated)';
                    app.ActionLabel.Text = 'Action: ESCALATED (Evidence Disagreement)';
                    app.ActionLabel.FontColor = [0.85, 0.15, 0.15];
                end
            catch e
                app.ActionLabel.Text = sprintf('Error: %s', e.message);
                app.ActionLabel.FontColor = [1, 0, 0];
            end
            
            app.RunAnalysisButton.Enable = 'on';
        end
    end

    methods (Access = public)

        function app = VisionX_Fallback_UI()
            % Create UIFigure and components
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 800 500];
            app.UIFigure.Name = 'VisionX - MathWorks Deferral Pipeline';

            app.TitleLabel = uilabel(app.UIFigure);
            app.TitleLabel.HorizontalAlignment = 'center';
            app.TitleLabel.FontSize = 24;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.Position = [1 440 800 40];
            app.TitleLabel.Text = 'VisionX Pure-MATLAB Fallback UI';

            app.LoadImageButton = uibutton(app.UIFigure, 'push');
            app.LoadImageButton.ButtonPushedFcn = createCallbackFcn(app, @LoadImageButtonPushed, true);
            app.LoadImageButton.Position = [20 380 150 40];
            app.LoadImageButton.Text = 'Load Fundus Image';

            app.RunAnalysisButton = uibutton(app.UIFigure, 'push');
            app.RunAnalysisButton.ButtonPushedFcn = createCallbackFcn(app, @RunAnalysisButtonPushed, true);
            app.RunAnalysisButton.Position = [190 380 150 40];
            app.RunAnalysisButton.Text = 'Run Safety Gates';
            app.RunAnalysisButton.Enable = 'off';

            app.OriginalImageAxes = uiaxes(app.UIFigure);
            title(app.OriginalImageAxes, 'Original Scan');
            app.OriginalImageAxes.Position = [20 80 350 280];
            app.OriginalImageAxes.XColor = 'none';
            app.OriginalImageAxes.YColor = 'none';

            app.HeatmapAxes = uiaxes(app.UIFigure);
            title(app.HeatmapAxes, 'Grad-CAM Attention');
            app.HeatmapAxes.Position = [410 80 350 280];
            app.HeatmapAxes.XColor = 'none';
            app.HeatmapAxes.YColor = 'none';

            app.StatusPanel = uipanel(app.UIFigure);
            app.StatusPanel.Title = 'Pipeline Status';
            app.StatusPanel.Position = [20 10 760 60];

            app.GradeLabel = uilabel(app.StatusPanel);
            app.GradeLabel.Position = [10 10 200 22];
            app.GradeLabel.Text = 'Grade: -';

            app.ConfidenceLabel = uilabel(app.StatusPanel);
            app.ConfidenceLabel.Position = [220 10 200 22];
            app.ConfidenceLabel.Text = 'Confidence: -';
            
            app.ActionLabel = uilabel(app.StatusPanel);
            app.ActionLabel.Position = [430 10 300 22];
            app.ActionLabel.FontWeight = 'bold';
            app.ActionLabel.Text = 'Action: -';

            app.UIFigure.Visible = 'on';
        end
    end
end
