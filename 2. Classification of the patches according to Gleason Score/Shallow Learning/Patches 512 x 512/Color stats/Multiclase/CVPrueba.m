clc; 
clearvars -global;
clear all;
%Base de Datos
load('Color_FSelection.mat','nca_Color');
rng('default');

Stats_path = fullfile(pwd,'Colorstats_Data 2.txt');
Stats_table = readtable(Stats_path);

% Cantidad de clases

cl = 4;

% Cantidad de caracteristicas
caract = size(Stats_table);
caract = caract(2) - 2;

% Separacion de Clases.
G0 = table2array(Stats_table(1:1000,2:end-1));
G3 = table2array(Stats_table(1001:2000,2:end-1));
G4 = table2array(Stats_table(2001:3000,2:end-1));
G5 = table2array(Stats_table(3001:4000,2:end-1));

DATOS = [G0;G3;G4;G5];

tp=size(DATOS);
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];
k=5;%127
d=caract*0.72;
tol = 0.1;
selidx = find(nca_Color.FeatureWeights > tol*max(1,max(nca_Color.FeatureWeights)));
% DATOS=DATOS(:,selidx);

% G0 = DATOS(1:size(G0, 1),:);
% G3 = DATOS(size(G0, 1)+1:size(G0, 1)+size(G3, 1),:);
% G4 = DATOS(size(G0, 1)+size(G3, 1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1),:);
% G5 = DATOS(size(G0, 1)+size(G3, 1)+size(G4,1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1)+size(G5,1),:);

% G0 = G0(randperm(size(G0, 1)), :);
% G3 = G3(randperm(size(G3, 1)), :);
% G4 = G4(randperm(size(G4, 1)), :);
% G5 = G5(randperm(size(G5, 1)), :);

caract=size(G3,2);

% Cantidad de Filas de la tabla
Filas = [size(G0, 1),size(G3, 1),size(G4, 1),size(G4, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;

% cvp = cvpartition(Y,'holdout',0.3);
% 
% XTrain3 = DATOS(cvp.training,:);
% YTrain3 = Y(cvp.training,:);
% 
% XTest3  = DATOS(cvp.test,:);
% YTest3  = Y(cvp.test,:);

% Datos de entrenamiento
XTrain3 = [G0(1:data_train,1:caract);G3(1:data_train,1:caract);G4(1:data_train,1:caract);G5(1:data_train,1:caract)];
YTrain3 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G3'}, data_train, 1));string(repmat({'G4'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];
%Datos Test
XTest3 = [G0(data_train+1:min(Filas),1:caract);G3(data_train+1:min(Filas),1:caract);G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTest3 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G3'}, data_test, 1));string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

kf = 10;

NNModelCV = fitcnet(XTrain3, YTrain3,'CrossVal','on','KFold',kf, 'Activations', 'relu', 'Standardize', true, 'LayerSizes', 292, 'Lambda', 0.00303415);

for fold = 1:kf
    % Realizar predicciones en los datos de prueba
    YPredCV = predict(NNModelCV.Trained{fold,1}, XTest3);
    % Puedes evaluar el rendimiento del modelo aquí si es necesario
    C3 = confusionmat(YTest3, YPredCV);

    % Calcular las métricas de desempeño
    accuracy = sum(diag(C3)) / sum(C3(:));
    
    title = "Clasificación Multiclase NN -" +  string(fold) + "-Fold. Accuracy = " + string(accuracy *100) + "%";
    
    YTest3 = cellstr(YTest3);
    
    figure
    cm3 = confusionchart(YTest3,YPredCV, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized',...
        'Title',title);
end
