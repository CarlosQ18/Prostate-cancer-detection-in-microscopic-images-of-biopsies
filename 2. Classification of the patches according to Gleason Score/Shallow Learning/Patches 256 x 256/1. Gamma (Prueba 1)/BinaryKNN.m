clc; 
clearvars -global;
clear all;
%Base de Datos
rng('default');
Stats_path = fullfile(pwd,'Gamma_GLCMstats_Train.txt');
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

%Eliminando outliers.
% G0 =rmoutliers(G0,'quartiles');
% G3 = rmoutliers(G3,'quartiles');
% G4 = rmoutliers(G4,'quartiles');
% G5 = rmoutliers(G5,'quartiles');

DATOS = [G0;G3;G4;G5];

mu = mean(DATOS);
sigma = std(DATOS);

% Estandariza los datos de entrenamiento
DATOS = (DATOS - mu) ./ sigma;

tp=size(DATOS);
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];
k=5;%127
d=caract*0.5;
[idx,w]=relieff(DATOS,Y,k);
DATOS=DATOS(:,idx(:,1:d));

G0 = DATOS(1:size(G0, 1),:);
G3 = DATOS(size(G0, 1)+1:size(G0, 1)+size(G3, 1),:);
G4 = DATOS(size(G0, 1)+size(G3, 1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1),:);
G5 = DATOS(size(G0, 1)+size(G3, 1)+size(G4,1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1)+size(G5,1),:);

G345 =[G3;G4;G5];
caract=size(G0,2);

% Cantidad de Filas de la tabla
Filas = [size(G0, 1),size(G345, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;
% Crear matriz de salida
class=zeros(cl,data_test*cl);

% Datos de entrenamiento
XTrain = [G0(1:data_train,1:caract);G3(1:(data_train/3) + 1,1:caract);G4(1:(data_train/3) + 1,1:caract);G5(1:data_train/3,1:caract)];
mu = mean(XTrain);
sigma = std(XTrain);
XTrain = (XTrain - mu) ./ sigma;

Y_TRAIN = [string(repmat({'G0'}, data_train, 1));string(repmat({'G345'}, data_train, 1))];

% Datos de testeo
XTest = [G0(data_train+1:min(Filas),1:caract);G3((data_train/3) + 2: (data_test/3) + (data_train/3) + 1,1:caract);G4((data_train/3) + 2:(data_test/3) + (data_train/3) + 1,1:caract);G5((data_train/3) + 1:(data_test/3) + (data_train/3),1:caract)];
mu = mean(XTest);
sigma = std(XTest);
XTest = (XTest - mu) ./ sigma;

Y_TEST = [string(repmat({'G0'}, data_test, 1));string(repmat({'G345'}, data_test, 1))];

% Definir las probabilidades a priori personalizadas
priorProb = [0.5,0.5]; % Probabilidades a priori para cada clase (suman 1)

classNames = {'G0','G345'};
Kfld = 5;

%results = fitcknn(XTrain, Y_TRAIN, 'ClassNames',classNames,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,'Kfold',Kfld));

dist = 'euclidean';
Neigh = 50;

KNNModel = fitcknn(XTrain, Y_TRAIN, 'ClassNames',classNames,'Distance',dist,'NumNeighbors',Neigh,'Prior', priorProb);

% Realizar predicciones en los datos de prueba
YPred = predict(KNNModel, XTest);

% % Generar la matriz de confusión
C = confusionmat(Y_TEST, YPred);

Y_TEST = cellstr(Y_TEST);

% Calcular las métricas de desempeño
accuracy = sum(diag(C)) / sum(C(:));

title = "Clasificación Binaria Grado 0 vs Grado 3-4-5 KNN " + string(accuracy);
figure
cm = confusionchart(Y_TEST,YPred, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

% Precisión
precision = diag(C) ./ sum(C, 1)';

% Recall
recall = diag(C) ./ sum(C, 2);

% F1-score
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Mostrar las métricas
fprintf('Exactitud: %.2f%%\n', accuracy * 100);
fprintf('Precisión: %.2f%%\n', mean(precision) * 100);
fprintf('Recall: %.2f%%\n', mean(recall) * 100);
fprintf('F1-score: %.2f%%\n', mean(f1score) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
G45 =[G4;G5];
caract=size(G3,2);

% Cantidad de Filas de la tabla
Filas = [size(G3, 1),size(G45, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;
% Crear matriz de salida
class=zeros(cl,data_test*cl);

% Datos de entrenamiento
XTrain2 = [G3(1:data_train,1:caract);G4(1:(data_train/2),1:caract);G5(1:data_train/2,1:caract)];
mu = mean(XTrain2);
sigma = std(XTrain2);
XTrain2 = (XTrain2 - mu) ./ sigma;

Y_TRAIN2 = [string(repmat({'G3'}, data_train, 1));string(repmat({'G45'}, data_train, 1))];

% Datos de testeo
XTest2 = [G3(data_train+1:min(Filas),1:caract);G4((data_train/2)+1:(data_test/2) + (data_train/2),1:caract);G5((data_train/2) + 1:(data_test/2) + (data_train/2),1:caract)];
mu = mean(XTest2);
sigma = std(XTest2);
XTest2 = (XTest2 - mu) ./ sigma;

Y_TEST2 = [string(repmat({'G3'}, data_test, 1));string(repmat({'G45'}, data_test, 1))];

% Definir las probabilidades a priori personalizadas
priorProb = [0.5,0.5]; % Probabilidades a priori para cada clase (suman 1)
classNames = {'G3','G45'};
Kfld = 5;

results = fitcknn(XTrain2, Y_TRAIN2, 'ClassNames',classNames,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,'Kfold',Kfld));

dist = results.Distance;
Neigh = results.NumNeighbors;

KNNModel2 = fitcknn(XTrain2, Y_TRAIN2, 'ClassNames',classNames,'Distance',dist,'NumNeighbors',Neigh,'Prior', priorProb);

% Realizar predicciones en los datos de prueba
YPred2 = predict(KNNModel2, XTest2);

% % Generar la matriz de confusión
C2 = confusionmat(Y_TEST2, YPred2);

% Calcular las métricas de desempeño
accuracy = sum(diag(C2)) / sum(C2(:));

Y_TEST2 = cellstr(Y_TEST2);

title = "Clasificación Binaria Grado 3 vs Grado 4-5 KNN " + string(accuracy);
figure
cm2 = confusionchart(Y_TEST2,YPred2, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);
% Calcular las métricas de desempeño
accuracy = sum(diag(C2)) / sum(C2(:));

% Precisión
precision = diag(C2) ./ sum(C2, 1)';

% Recall
recall = diag(C2) ./ sum(C2, 2);

% F1-score
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Mostrar las métricas
fprintf('Exactitud: %.2f%%\n', accuracy * 100);
fprintf('Precisión: %.2f%%\n', mean(precision) * 100);
fprintf('Recall: %.2f%%\n', mean(recall) * 100);
fprintf('F1-score: %.2f%%\n', mean(f1score) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
caract=size(G4,2);

% Cantidad de Filas de la tabla
Filas = [size(G4, 1),size(G5, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;
% Crear matriz de salida
class=zeros(cl,data_test*cl);

% Datos de entrenamiento
XTrain3 = [G4(1:data_train,1:caract);G5(1:data_train,1:caract)];
mu = mean(XTrain3);
sigma = std(XTrain3);
XTrain3 = (XTrain3 - mu) ./ sigma;

Y_TRAIN3 = [string(repmat({'G4'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];

% Datos de testeo
XTest3 = [G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
mu = mean(XTest3);
sigma = std(XTest3);
XTest3 = (XTest3 - mu) ./ sigma;

Y_TEST3 = [string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Definir las probabilidades a priori personalizadas
priorProb = [0.5,0.5]; % Probabilidades a priori para cada clase (suman 1)
classNames = {'G4','G5'};
Kfld = 5;

results = fitcknn(XTrain3, Y_TRAIN3, 'ClassNames',classNames,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,'Kfold',Kfld));

dist = results.Distance;
Neigh = results.NumNeighbors;

KNNModel3 = fitcknn(XTrain3, Y_TRAIN3, 'ClassNames',classNames,'Distance',dist,'NumNeighbors',Neigh,'Prior', priorProb);
% Realizar predicciones en los datos de prueba
YPred3 = predict(KNNModel3, XTest3);

% % Generar la matriz de confusión
C3 = confusionmat(Y_TEST3, YPred3);

% Calcular las métricas de desempeño
accuracy = sum(diag(C3)) / sum(C3(:));

title = "Clasificación Binaria Grado 4 vs Grado 5 KNN " + string(accuracy);

Y_TEST3 = cellstr(Y_TEST3);

figure
cm3 = confusionchart(Y_TEST3,YPred3, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

% Precisión
precision = diag(C3) ./ sum(C3, 1)';

% Recall
recall = diag(C3) ./ sum(C3, 2);

% F1-score
f1score = 2 * (precision .* recall) ./ (precision + recall);

% Mostrar las métricas
fprintf('Exactitud: %.2f%%\n', accuracy * 100);
fprintf('Precisión: %.2f%%\n', mean(precision) * 100);
fprintf('Recall: %.2f%%\n', mean(recall) * 100);
fprintf('F1-score: %.2f%%\n', mean(f1score) * 100);
