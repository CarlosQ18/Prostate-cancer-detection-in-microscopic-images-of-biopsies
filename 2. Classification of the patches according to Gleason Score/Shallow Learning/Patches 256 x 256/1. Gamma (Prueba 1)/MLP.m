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

% mu = mean(DATOS);
% sigma = std(DATOS);
% 
% % Estandariza los datos de entrenamiento
% DATOS = (DATOS - mu) ./ sigma;

tp=size(DATOS);
Y=[string(repmat({'G0'}, size(G0, 1), 1));string(repmat({'G3'}, size(G3, 1), 1));string(repmat({'G4'}, size(G4, 1), 1));string(repmat({'G5'}, size(G5, 1), 1))];
k=5;%127
d=caract*0.5;;
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
XTrain = XTrain';

Y_TRAIN = [ones(1,data_train), zeros(1,data_train); zeros(1,data_train), ones(1,data_train)];

% Datos de testeo
XTest = [G0(data_train+1:min(Filas),1:caract);G3((data_train/3) + 2: (data_test/3) + (data_train/3) + 1,1:caract);G4((data_train/3) + 2:(data_test/3) + (data_train/3) + 1,1:caract);G5((data_train/3) + 1:(data_test/3) + (data_train/3),1:caract)];
mu = mean(XTest);
sigma = std(XTest);
XTest = (XTest - mu) ./ sigma;
XTest = XTest';

Y_TEST = [ones(1,data_test), zeros(1,data_test); zeros(1,data_test), ones(1,data_test)];

net =  patternnet(4);
net.layers{1}.transferFcn='poslin'; %con esta línea se cambia la función de activación de la primera capa a ReLU
net.trainFcn = 'trainscg';
%net.layers{2}.transferFcn='logsig';
%net.trainParam.lr = 0.001;
[net, tr] = train(net,XTrain,Y_TRAIN);
% figure;
% plotperform(tr);

ytest = net(XTest);

% figure;
% plotconfusion(Y_TEST,ytest);
% perftest = perform(net,XTest,ytest);

Y_TEST = vec2ind(Y_TEST);
YPred = vec2ind(ytest);

C = confusionmat(Y_TEST, YPred);
accuracy = sum(diag(C)) / sum(C(:));

figure

title = "Clasificación Binaria Grado 0 vs Grado 3-4-5 MLP " + string(accuracy);

confusionchart(C,{'G0','G345'},...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
XTrain2 = XTrain2';

Y_TRAIN2 = [ones(1,data_train), zeros(1,data_train); zeros(1,data_train), ones(1,data_train)];

% Datos de testeo
XTest2 = [G3(data_train+1:min(Filas),1:caract);G4((data_train/2)+1:(data_test/2) + (data_train/2),1:caract);G5((data_train/2) + 1:(data_test/2) + (data_train/2),1:caract)];
mu = mean(XTest2);
sigma = std(XTest2);
XTest2 = (XTest2 - mu) ./ sigma;
XTest2 = XTest2';

Y_TEST2 = [ones(1,data_test), zeros(1,data_test); zeros(1,data_test), ones(1,data_test)];


net2 =  patternnet(4);
net2.layers{1}.transferFcn='poslin'; %con esta línea se cambia la función de activación de la primera capa a ReLU
net2.trainFcn = 'trainlm';
%net.layers{2}.transferFcn='logsig';
%net.trainParam.lr = 0.001;
[net2, tr2] = train(net2,XTrain2,Y_TRAIN2);
% figure;
% plotperform(tr);

ytest2 = net2(XTest2);

% figure;
% plotconfusion(Y_TEST,ytest);
% perftest = perform(net,XTest,ytest);

Y_TEST2 = vec2ind(Y_TEST2);
YPred2 = vec2ind(ytest2);

C2 = confusionmat(Y_TEST2, YPred2);
accuracy = sum(diag(C2)) / sum(C2(:));

figure

title = "Clasificación Binaria Grado 3 vs Grado 4-5 MLP " + string(accuracy);

confusionchart(C2,{'G3','G45'},...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
XTrain3 = XTrain3';

Y_TRAIN3 = [ones(1,data_train), zeros(1,data_train); zeros(1,data_train), ones(1,data_train)];


% Datos de testeo
XTest3 = [G4(data_train+1:min(Filas),1:caract);G5(data_train+1:min(Filas),1:caract)];
mu = mean(XTest3);
sigma = std(XTest3);
XTest3 = (XTest3 - mu) ./ sigma;
XTest3 = XTest3';

Y_TEST3 = [ones(1,data_test), zeros(1,data_test); zeros(1,data_test), ones(1,data_test)];


net3 =  patternnet(292);
net3.layers{1}.transferFcn='satlin'; %con esta línea se cambia la función de activación de la primera capa a ReLU
net3.trainFcn = 'trainscg';
%net.layers{2}.transferFcn='logsig';
%net.trainParam.lr = 0.001;
[net3, tr3] = train(net3,XTrain3,Y_TRAIN3);
% figure;
% plotperform(tr);

ytest3 = net3(XTest3);

% figure;
% plotconfusion(Y_TEST,ytest);
% perftest = perform(net,XTest,ytest);

Y_TEST3 = vec2ind(Y_TEST3);
YPred3 = vec2ind(ytest3);

C3 = confusionmat(Y_TEST3, YPred3);
accuracy = sum(diag(C3)) / sum(C3(:));

figure

title = "Clasificación Binaria Grado 4 vs Grado 5 MLP " + string(accuracy);

confusionchart(C3,{'G4','G5'},...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized',...
    'Title',title);
