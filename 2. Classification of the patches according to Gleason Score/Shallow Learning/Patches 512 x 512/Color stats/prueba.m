clc; 
clearvars -global;
clear all;
%Base de Datos

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

%Eliminando outliers.
% G0 =rmoutliers(G0,'quartiles');
% G3 = rmoutliers(G3,'quartiles');
% G4 = rmoutliers(G4,'quartiles');
% G5 = rmoutliers(G5,'quartiles');

DATOS = [G0;G3;G4;G5];

% mu = mean(DATOS);
% sigma = std(DATOS);
% 
% % Estandariza los datos de entrenamiento
% DATOS = (DATOS - mu) ./ sigma;

tp=size(DATOS);
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];
k=5;%127
d=15;
[idx,w]=relieff(DATOS,Y,k);
DATOS=DATOS(:,idx(:,1:d));

G0 = DATOS(1:size(G0, 1),:);
G3 = DATOS(size(G0, 1)+1:size(G0, 1)+size(G3, 1),:);
G4 = DATOS(size(G0, 1)+size(G3, 1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1),:);
G5 = DATOS(size(G0, 1)+size(G3, 1)+size(G4,1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1)+size(G5,1),:);

caract=size(G3,2);

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
mu1 = mean(XTrain3);
sigma1 = std(XTrain3);
XTrain3 = (XTrain3 - mu1) ./ sigma1;

Y_TRAIN3 = [string(repmat({'G4'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];

% Datos de testeo
XTest3 = [G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
% mu2 = mean(XTest3);
% sigma2 = std(XTest3);
XTest3Stand = (XTest3 - mu1) ./ sigma1;

Y_TEST3 = [string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Definir las probabilidades a priori personalizadas
priorProb = [0.5,0.5]; % Probabilidades a priori para cada clase (suman 1)
classNames = {'G4','G5'};
Kfld = 5;

NNModel = fitcsvm(XTrain3, Y_TRAIN3, 'ClassNames',classNames,'Standardize',false,'KernelFunction','linear','Prior', priorProb);

% Realizar predicciones en los datos de prueba
YPred = predict(NNModel, XTest3Stand);

% % Generar la matriz de confusión
C = confusionmat(Y_TEST3, YPred);

Y_TEST3 = cellstr(Y_TEST3);

% Calcular las métricas de desempeño
accuracy = sum(diag(C)) / sum(C(:));

title = "Clasificación Binaria Grado 0 vs Grado 3 NN " + string(accuracy);
figure
cm = confusionchart(Y_TEST3,YPred, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

