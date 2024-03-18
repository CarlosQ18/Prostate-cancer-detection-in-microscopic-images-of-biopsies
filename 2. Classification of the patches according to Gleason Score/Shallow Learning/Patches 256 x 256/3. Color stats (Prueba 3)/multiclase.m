clear all;
clc;

rng('default');

Stats_path = fullfile(pwd,'Colorstats_Train.txt');
Stats_table = readtable(Stats_path);

%Stats_table(:, [5]) = []; % Aquellas que utilizan GLCM prueba 1 (Gamma)y prueba 3 (Filtros PA) 
%Stats_table(:, [5,27,49,71,93]) = [] % Aquellas que utilizan GLCM prueba 5 (Wavelet)

% Cantidad de clases

cl = 4;

% Cantidad de caracteristicas
caract = size(Stats_table);
caract = caract(2) - 2;

% Separacion de Clases.
G0 = table2array(Stats_table(1:5000,2:end-1));
G3 = table2array(Stats_table(5001:10000,2:end-1));
G4 = table2array(Stats_table(10001:15000,2:end-1));
G5 = table2array(Stats_table(15001:20000,2:end-1));

DATOS = [G0;G3;G4;G5];
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];

k=5;%127
d=caract*0.5;
[idx,w]=relieff(DATOS,Y,k);
DATOS=DATOS(:,idx(:,1:d));

particion = 0.2;

cvp = cvpartition(Y,'holdout',0.2,'Stratify',true);

XTrain = DATOS(cvp.training,:);
YTrain = Y(cvp.training,:);

XTest  = DATOS(cvp.test,:);
YTest  = Y(cvp.test,:);

classNames = {'G0','G3','G4','G5'};
Kfld = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NAIVE BAYES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Definir las probabilidades a priori personalizadas
priorProb = [0.25,0.25,0.25,0.25]; % Probabilidades a priori para cada clase (suman 1)

% NBModel = fitcnb(XTrain, YTrain, 'ClassNames',classNames,'OptimizeHyperparameters','all',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,'Kfold',Kfld));
% 
% Distrib = NBModel.DistributionNames; %% 'kernel'
% Wdth = NBModel.Width; %% 0.110925584083734
% krnl = NBModel.Kernel; %% 'normal'

NBModel = fitcnb(XTrain, YTrain, 'ClassNames',classNames, 'DistributionNames','kernel','Width',0.110925584083734,'Kernel','normal','Standardize',true);
% Realizar predicciones en los datos de prueba
YPred = predict(NBModel, XTest);

% % Generar la matriz de confusión
C = confusionmat(YTest, YPred);

Y_TEST = cellstr(YTest);

% Calcular las métricas de desempeño
accuracy = sum(diag(C)) / sum(C(:));

title = "Clasificación MULTICLASE Bayes " + string(accuracy);
figure
cm = confusionchart(Y_TEST,YPred, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t2 = templateKNN('Standardize',true,'Distance','euclidean','NumNeighbors',24);
KNNModel = fitcecoc(XTrain,YTrain,'ClassNames',classNames,'Prior','uniform','Learners',t2,'Coding','onevsone');

% Realizar predicciones en los datos de prueba
YPredKNN = predict(KNNModel, XTest);

C2 = confusionmat(YTest, YPredKNN);

% Calcular las métricas de desempeño
accuracy = sum(diag(C2)) / sum(C2(:));

title = "Clasificación Multiclase KNN "+ string(accuracy);

Y_TEST = cellstr(YTest);

figure
cm2 = confusionchart(Y_TEST,YPredKNN, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%% SVM %%%%%%%%%%%%%%%%%%%%%%%%

t = templateSVM('Standardize',true,'KernelFunction','rbf',"Type","classification",'Solver','ISDA');
SVMModel = fitcecoc(XTrain,YTrain,'ClassNames',classNames,'Prior','uniform','Learners',t,'Coding','onevsone');
% Realizar predicciones en los datos de prueba
YPredSVM = predict(SVMModel, XTest);

C3 = confusionmat(YTest, YPredSVM);

% Calcular las métricas de desempeño
accuracy = sum(diag(C3)) / sum(C3(:));

title = "Clasificación Multiclase SVM "+ string(accuracy);

Y_TEST = cellstr(YTest);

figure
cm3 = confusionchart(Y_TEST,YPredSVM, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%% MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NNModel = fitcnet(XTrain,YTrain,"Activations","relu","Standardize",true,"LayerSizes",292,'Lambda',0.0000704);
% % Realizar predicciones en los datos de prueba
YPredNN = predict(NNModel, XTest);


C4 = confusionmat(YTest, YPredNN);

% Calcular las métricas de desempeño
accuracy = sum(diag(C4)) / sum(C4(:));

title = "Clasificación Multiclase MLP " + string(accuracy);

Y_TEST = cellstr(YTest);

figure
cm4 = confusionchart(Y_TEST,YPredNN, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);