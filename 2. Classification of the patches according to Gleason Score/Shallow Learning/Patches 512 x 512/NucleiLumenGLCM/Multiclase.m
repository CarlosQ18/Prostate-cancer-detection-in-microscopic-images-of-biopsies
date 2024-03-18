clc; 
clearvars -global;
clear all;
%Base de Datos
load('nca_LumenNuclei_GLCMstats.mat','nca_LumenNuclei_GLCMstats');
rng('default');

Stats_path = fullfile(pwd,'LumenNuclei_GLCMstats.txt');
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

% k=5;%127
% d=caract*0.72;
% [idx,w]=relieff(DATOS,Y,k);
% DATOS=DATOS(:,idx(:,1:d));

tol    = 0.01;
selidx = find(nca_LumenNuclei_GLCMstats.FeatureWeights > tol*max(1,max(nca_LumenNuclei_GLCMstats.FeatureWeights)));
DATOS=DATOS(:,selidx);

G0 = G0(randperm(size(G0, 1)), :);
G3 = G3(randperm(size(G3, 1)), :);
G4 = G4(randperm(size(G4, 1)), :);
G5 = G5(randperm(size(G5, 1)), :);

% G0 = DATOS(1:size(G0, 1),:);
% G3 = DATOS(size(G0, 1)+1:size(G0, 1)+size(G3, 1),:);
% G4 = DATOS(size(G0, 1)+size(G3, 1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1),:);
% G5 = DATOS(size(G0, 1)+size(G3, 1)+size(G4,1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1)+size(G5,1),:);

caract=size(G3,2);

% Cantidad de Filas de la tabla
Filas = [size(G0, 1),size(G3, 1),size(G4, 1),size(G4, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;

% Datos de entrenamiento
XTrain3 = [G0(1:data_train,1:caract);G3(1:data_train,1:caract);G4(1:data_train,1:caract);G5(1:data_train,1:caract)];
YTrain3 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G3'}, data_train, 1));string(repmat({'G4'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];
%Datos Test
XTest3 = [G0(data_train+1:min(Filas),1:caract);G3(data_train+1:min(Filas),1:caract);G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTest3 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G3'}, data_test, 1));string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Definir las probabilidades a priori personalizadas
priorProb = [0.5,0.5]; % Probabilidades a priori para cada clase (suman 1)
classNames = {'G0','G3','G4','G5'};
Kernels = {'linear','gaussian','rbf','polynomial'};
Cod = {'onevsone','allpairs','binarycomplete','denserandom','onevsall','ordinal','sparserandom','ternarycomplete' };
Opti = {'ISDA','L1QP','SMO'}
% for i = 1:length(Opti)
    t = templateSVM('Standardize',true,'KernelFunction','linear',"Type","classification","Solver",'ISDA');
    SVMModel = fitcecoc(XTrain3,YTrain3,'ClassNames',classNames,'Prior','uniform','Learners',t,'Coding','onevsone');
    % Realizar predicciones en los datos de prueba
    YPred = predict(SVMModel, XTest3);

    C = confusionmat(YTest3, YPred);

    % Calcular las métricas de desempeño
    accuracy = sum(diag(C)) / sum(C(:));

    title = "Clasificación Multiclase SVM "+ string(accuracy);

    YTest3 = cellstr(YTest3);

    figure
    cm = confusionchart(YTest3,YPred, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized',...
        'Title',title);
% end



NNModel3 = fitcnet(XTrain3,YTrain3,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415)
% Realizar predicciones en los datos de prueba
YPred3 = predict(NNModel3, XTest3);

C3 = confusionmat(YTest3, YPred3);

% Calcular las métricas de desempeño
accuracy = sum(diag(C3)) / sum(C3(:));

title = "Clasificación Multiclase NN " + string(accuracy);

YTest3 = cellstr(YTest3);

figure
cm3 = confusionchart(YTest3,YPred3, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);