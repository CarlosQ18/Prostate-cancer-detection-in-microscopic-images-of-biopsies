clear all;
clc;

rng('default');

Stats_path = fullfile(pwd,'Wavelet_GLCMstats_Train.txt');
Stats_table = readtable(Stats_path);

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
%[idx,w]=relieff(DATOS,Y,k);
idx = [42	25	26	49	4 3	20	33	5	32	27	50	64	59	40	23	43	69	70	106 45	62	57	29	1	11	10	37	44	63	48	47	114 113 61	46	126 58	39	24	93	21	104	 15	119	 31	66	128 7	18	38	22	41	92	91	51	97	28	17	2	35	60	36	71	65	53	99	98	109	 130 95	110	19 105	 90	121 6 84 120 116 68 83	77	88	13	76	73	87	54	9  129  56  55 34 14	 16	118	 131 115 117 127 112 111 132 108	82 122	125	103	 89 100	30 75 12 124 123 52 101	102	86 94 81 67 72 78 79 96 80	8 107 85	74];
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

t2 = templateKNN('Standardize',true,'Distance','chebychev','NumNeighbors',14);
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

t = templateSVM('Standardize',true,'KernelFunction','linear',"Type","classification",'Solver','ISDA');
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