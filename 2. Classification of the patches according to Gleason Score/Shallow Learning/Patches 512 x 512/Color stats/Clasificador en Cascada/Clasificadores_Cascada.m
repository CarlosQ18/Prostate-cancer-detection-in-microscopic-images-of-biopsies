clc; 
clearvars -global;
clear all;

rng("default");

%Base de Datos
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
d=caract*0.7;
[idx,w]=relieff(DATOS,Y,k);
DATOS=DATOS(:,idx(:,1:d));

G0 = DATOS(1:size(G0, 1),:);
% G01 = G0(randperm(size(G0, 1)), :);

G3 = DATOS(size(G0, 1)+1:size(G0, 1)+size(G3, 1),:);
% G3 = G3(randperm(size(G3, 1)), :);

G4 = DATOS(size(G0, 1)+size(G3, 1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1),:);
% G4 = G4(randperm(size(G4, 1)), :);

G5 = DATOS(size(G0, 1)+size(G3, 1)+size(G4,1)+1:size(G0, 1)+size(G3, 1)+size(G4, 1)+size(G5,1),:);
% G5 = G5(randperm(size(G5, 1)), :);

% Cantidad de Filas de la tabla
Filas = [size(G0, 1),size(G3, 1),size(G4, 1),size(G5, 1)];
% Cantidad de datos de entrenamiento
data_train = 0.7*min(Filas);
data_train = round(data_train);
% Cantidad de datos de testeo
data_test = min(Filas)- data_train;

caract = d;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 0 VS GRADO 3-4-5  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

G345 =[G3;G4;G5];

% Datos de entrenamiento G0 VS G345
XTrain_G0_vs_G345 = [G0(1:data_train,1:caract);G3(1:(data_train/3),1:caract);G4(1:(data_train/3),1:caract);G5(1:(data_train/3)+1,1:caract)];
YTrain_G0_vs_G345 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G345'}, data_train, 1))];

% Datos de testeo G0 VS G345
XTest_G0_vs_G345 = [G0(data_train+1:min(Filas),1:caract);G3((data_train/3) + 2: (data_test/3) + (data_train/3) + 1,1:caract);G4((data_train/3) + 2:(data_test/3) + (data_train/3) + 1,1:caract);G5((data_train/3) + 1:(data_test/3) + (data_train/3),1:caract)];
YTest_G0_vs_G345 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G345'}, data_test, 1))];

% Entrenando el modelo
NNModel_G0_vs_G345 = fitcnet(XTrain_G0_vs_G345,YTrain_G0_vs_G345,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G0_vs_G345 = predict(NNModel_G0_vs_G345, XTest_G0_vs_G345);

% % Generar la matriz de confusión
C_G0_vs_G345 = confusionmat(YTest_G0_vs_G345, YPred_G0_vs_G345);

YTest_G0_vs_G345 = cellstr(YTest_G0_vs_G345);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G0_vs_G345)) / sum(C_G0_vs_G345(:));

title = "Clasificación Binaria Grado 0 vs Grado 3-4-5 NN " + string(accuracy);
figure
cm_G0_vs_G345 = confusionchart(YTest_G0_vs_G345,YPred_G0_vs_G345, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 0 VS GRADO 3  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G0 vs G3
XTrain_G0_vs_G3 = [G0(1:data_train,1:caract);G3(1:data_train,1:caract)];
YTrain_G0_vs_G3 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G3'}, data_train, 1))];

% Datos de testeo G0 vs G3
XTest_G0_vs_G3 = [G0(data_train+1:min(Filas),1:caract);G3(data_train+1:min(Filas),1:caract)];
YTest_G0_vs_G3 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G3'}, data_test, 1))];

% Entrenando el modelo
NNModel_G0_vs_G3 = fitcnet(XTrain_G0_vs_G3,YTrain_G0_vs_G3,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G0_vs_G3 = predict(NNModel_G0_vs_G3, XTest_G0_vs_G3);

% % Generar la matriz de confusión
C_G0_vs_G3 = confusionmat(YTest_G0_vs_G3, YPred_G0_vs_G3);

YTest_G0_vs_G3 = cellstr(YTest_G0_vs_G3);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G0_vs_G3)) / sum(C_G0_vs_G3(:));

title = "Clasificación Binaria Grado 0 vs Grado 3 NN " + string(accuracy);
figure
cm_G0_vs_G3 = confusionchart(YTest_G0_vs_G3,YPred_G0_vs_G3, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 0 VS GRADO 4  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G0 vs G4
XTrain_G0_vs_G4 = [G0(1:data_train,1:caract);G4(1:data_train,1:caract)];
YTrain_G0_vs_G4 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G4'}, data_train, 1))];

% Datos de testeo G0 vs G4
XTest_G0_vs_G4 = [G0(data_train+1:min(Filas),1:caract);G4(data_train+1:min(Filas),1:caract)];
YTest_G0_vs_G4 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G4'}, data_test, 1))];

% Entrenando el modelo
NNModel_G0_vs_G4 = fitcnet(XTrain_G0_vs_G4,YTrain_G0_vs_G4,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G0_vs_G4 = predict(NNModel_G0_vs_G4, XTest_G0_vs_G4);

% % Generar la matriz de confusión
C_G0_vs_G4 = confusionmat(YTest_G0_vs_G4, YPred_G0_vs_G4);

YTest_G0_vs_G4 = cellstr(YTest_G0_vs_G4);
% Calcular las métricas de desempeño
accuracy = sum(diag(C_G0_vs_G4)) / sum(C_G0_vs_G4(:));

title = "Clasificación Binaria Grado 0 vs Grado 4 NN " + string(accuracy);
figure
cm_G0_vs_G4 = confusionchart(YTest_G0_vs_G4,YPred_G0_vs_G4, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 0 VS GRADO 5  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G0 vs G5
XTrain_G0_vs_G5 = [G0(1:data_train,1:caract);G5(1:data_train,1:caract)];
YTrain_G0_vs_G5 = [string(repmat({'G0'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];

% Datos de testeo G0 vs G5
XTest_G0_vs_G5 = [G0(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTest_G0_vs_G5 = [string(repmat({'G0'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Entrenando el modelo
NNModel_G0_vs_G5 = fitcnet(XTrain_G0_vs_G5,YTrain_G0_vs_G5,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G0_vs_G5 = predict(NNModel_G0_vs_G5, XTest_G0_vs_G5);

% % Generar la matriz de confusión
C_G0_vs_G5 = confusionmat(YTest_G0_vs_G5, YPred_G0_vs_G5);

YTest_G0_vs_G5 = cellstr(YTest_G0_vs_G5);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G0_vs_G5)) / sum(C_G0_vs_G5(:));

title = "Clasificación Binaria Grado 0 vs Grado 5 NN " + string(accuracy);
figure
cm_G0_vs_G5 = confusionchart(YTest_G0_vs_G5,YPred_G0_vs_G5, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   GRADO 3 VS GRADO 4-5  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G3 vs G45
XTrain_G3_vs_G45 = [G3(1:data_train,1:caract);G4(1:(data_train/2),1:caract);G5(1:data_train/2,1:caract)];
YTrain_G3_vs_G45 = [string(repmat({'G3'}, data_train, 1));string(repmat({'G45'}, data_train, 1))];

% Datos de testeo G3 vs G45
XTest_G3_vs_G45 = [G3(data_train+1:min(Filas),1:caract);G4((data_train/2)+1:(data_test/2) + (data_train/2),1:caract);G5((data_train/2) + 1:(data_test/2) + (data_train/2),1:caract)];
YTest_G3_vs_G45 = [string(repmat({'G3'}, data_test, 1));string(repmat({'G45'}, data_test, 1))];

% Entrenando el modelo
NNModel_G3_vs_G45 = fitcnet(XTrain_G3_vs_G45,YTrain_G3_vs_G45,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G3_vs_G45 = predict(NNModel_G3_vs_G45, XTest_G3_vs_G45);

% % Generar la matriz de confusión
C_G3_vs_G45 = confusionmat(YTest_G3_vs_G45, YPred_G3_vs_G45);

YTest_G3_vs_G45 = cellstr(YTest_G3_vs_G45);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G3_vs_G45)) / sum(C_G3_vs_G45(:));

title = "Clasificación Binaria Grado 3 vs Grado 4-5 NN " + string(accuracy);
figure
cm_G3_vs_G45 = confusionchart(YTest_G3_vs_G45,YPred_G3_vs_G45, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 3 VS GRADO 4  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G3 vs G4
XTrain_G3_vs_G4 = [G3(1:data_train,1:caract);G4(1:data_train,1:caract)];
YTrain_G3_vs_G4 = [string(repmat({'G3'}, data_train, 1));string(repmat({'G4'}, data_train, 1))];

% Datos de testeo G3 vs G4
XTest_G3_vs_G4 = [G3(data_train+1:min(Filas),1:caract);G4(data_train+1:min(Filas),1:caract)];
YTest_G3_vs_G4 = [string(repmat({'G3'}, data_test, 1));string(repmat({'G4'}, data_test, 1))];

% Entrenando el modelo
NNModel_G3_vs_G4 = fitcnet(XTrain_G3_vs_G4,YTrain_G3_vs_G4,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G3_vs_G4 = predict(NNModel_G3_vs_G4, XTest_G3_vs_G4);

% % Generar la matriz de confusión
C_G3_vs_G4 = confusionmat(YTest_G3_vs_G4, YPred_G3_vs_G4);

YTest_G3_vs_G4 = cellstr(YTest_G3_vs_G4);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G3_vs_G4)) / sum(C_G3_vs_G4(:));

title = "Clasificación Binaria Grado 3 vs Grado 4 NN " + string(accuracy);
figure
cm_G3_vs_G4 = confusionchart(YTest_G3_vs_G4,YPred_G3_vs_G4, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 3 VS GRADO 5  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G3 vs G5
XTrain_G3_vs_G5 = [G3(1:data_train,1:caract);G5(1:data_train,1:caract)];
YTrain_G3_vs_G5 = [string(repmat({'G3'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];

% Datos de testeo G3 vs G5
XTest_G3_vs_G5 = [G3(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTest_G3_vs_G5 = [string(repmat({'G3'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Entrenando el modelo
NNModel_G3_vs_G5 = fitcnet(XTrain_G3_vs_G5,YTrain_G3_vs_G5,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G3_vs_G5 = predict(NNModel_G3_vs_G5, XTest_G3_vs_G5);

% % Generar la matriz de confusión
C_G3_vs_G5 = confusionmat(YTest_G3_vs_G5, YPred_G3_vs_G5);

YTest_G3_vs_G5 = cellstr(YTest_G3_vs_G5);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G3_vs_G5)) / sum(C_G3_vs_G5(:));

title = "Clasificación Binaria Grado 3 vs Grado 5 NN " + string(accuracy);
figure
cm_G3_vs_G5 = confusionchart(YTest_G3_vs_G5,YPred_G3_vs_G5, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  GRADO 4 VS GRADO 5  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Datos de entrenamiento G4 vs G5
XTrain_G4_vs_G5 = [G4(1:data_train,1:caract);G5(1:data_train,1:caract)];
YTrain_G4_vs_G5 = [string(repmat({'G4'}, data_train, 1));string(repmat({'G5'}, data_train, 1))];

% Datos de testeo G4 vs G5
XTest_G4_vs_G5 = [G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTest_G4_vs_G5 = [string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];

% Entrenando el modelo
NNModel_G4_vs_G5 = fitcnet(XTrain_G4_vs_G5,YTrain_G4_vs_G5,"Activations","relu","Standardize",true,"LayerSizes",292,"Lambda",0.00303415);

% Realizar predicciones en los datos de prueba
YPred_G4_vs_G5 = predict(NNModel_G4_vs_G5, XTest_G4_vs_G5);

% % Generar la matriz de confusión
C_G4_vs_G5 = confusionmat(YTest_G4_vs_G5, YPred_G4_vs_G5);

YTest_G4_vs_G5 = cellstr(YTest_G4_vs_G5);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_G4_vs_G5)) / sum(C_G4_vs_G5(:));

title = "Clasificación Binaria Grado 4 vs Grado 5 NN " + string(accuracy);
figure
cm_G4_vs_G5 = confusionchart(YTest_G4_vs_G5,YPred_G4_vs_G5, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  EVALUACIÓN FINAL  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


XTestFinal = [G0(data_train+1:min(Filas),1:caract);G3(data_train+1:min(Filas),1:caract);G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
YTestFinal = [string(repmat({'G0'}, data_test, 1));string(repmat({'G3'}, data_test, 1));string(repmat({'G4'}, data_test, 1));string(repmat({'G5'}, data_test, 1))];
YPredFinal = [];

for i = 1 : length(XTestFinal)
    G0_vs_G345 = string(predict(NNModel_G0_vs_G345, XTestFinal(i,:)));
    G0_vs_G3 = predict(NNModel_G0_vs_G3, XTestFinal(i,:));
    G0_vs_G4 = predict(NNModel_G0_vs_G4, XTestFinal(i,:));
    G0_vs_G5 = predict(NNModel_G0_vs_G5, XTestFinal(i,:));
    if (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G0')
        YPredFinal = [YPredFinal;G0_vs_G345];
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G5')
        G3_vs_G5 = predict(NNModel_G3_vs_G5, XTestFinal(i,:));
        if (string(G3_vs_G5) == 'G5')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G5];
        end
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G0')
        G3_vs_G4 = predict(NNModel_G3_vs_G4, XTestFinal(i,:));
        YPredFinal = [YPredFinal;G3_vs_G4];
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G5')
        G3_vs_45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G0')
        YPredFinal = [YPredFinal;G0_vs_G3];
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G5')
        G3_vs_G5 = predict(NNModel_G3_vs_G5, XTestFinal(i,:));
        YPredFinal = [YPredFinal;G3_vs_G5];
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G0')
        G3_vs_G4 = predict(NNModel_G3_vs_G4, XTestFinal(i,:));
        YPredFinal = [YPredFinal;G3_vs_G4];
    elseif (string(G0_vs_G345) == 'G0' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G5')
        G3_vs_45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
    elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G0')
        YPredFinal = [YPredFinal;G0_vs_G5];
    elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G5')
        G3_vs_G5 = predict(NNModel_G3_vs_G5, XTestFinal(i,:));
        if (string(G3_vs_G5) == 'G5')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G5];
        end
    elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G0')
        G3_vs_G45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
    elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G0' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G5')
        G3_vs_G45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
   elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G0')
        G3_vs_G45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
   elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G0' && string(G0_vs_G5) == 'G5')
        G3_vs_G45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
   elseif(string(G0_vs_G345) == 'G345' && string(G0_vs_G3) == 'G3' && string(G0_vs_G4) == 'G4' && string(G0_vs_G5) == 'G0')
        G3_vs_G4 = predict(NNModel_G3_vs_G4, XTestFinal(i,:));
        YPredFinal = [YPredFinal;G3_vs_G4];
   else
        G3_vs_G45 = predict(NNModel_G3_vs_G45, XTestFinal(i,:));
        if (string(G3_vs_G45) == 'G45')
            G4_vs_G5 = predict(NNModel_G4_vs_G5, XTestFinal(i,:));
            YPredFinal = [YPredFinal;G4_vs_G5];
        else
            YPredFinal = [YPredFinal;G3_vs_G45];
        end
   end
end

% % Generar la matriz de confusión
C_Final = confusionmat(YTestFinal, YPredFinal);

% Calcular las métricas de desempeño
accuracy = sum(diag(C_Final)) / sum(C_Final(:));

title = "Clasificación en Cascada NN " + string(accuracy);
figure
cm_Final = confusionchart(YTestFinal,YPredFinal, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);
