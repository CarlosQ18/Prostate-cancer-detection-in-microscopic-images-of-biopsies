clc; 
clearvars -global;
clear all;

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

cvp = cvpartition(Y,'holdout',0.3);

Xtrain = DATOS(cvp.training,:);
ytrain = Y(cvp.training,:);
Xtest  = DATOS(cvp.test,:);
ytest  = Y(cvp.test,:);

% nca = fscnca(Xtrain,ytrain,'FitMethod','none');
% L = loss(nca,Xtest,ytest)
% 
% nca = fscnca(Xtrain,ytrain,'FitMethod','exact','Lambda',0,...
%       'Solver','lbfgs','Standardize',true);
% L = loss(nca,Xtest,ytest)

cvp = cvpartition(ytrain,'kfold',10);
numvalidsets = cvp.NumTestSets;

n = length(ytrain);
lambdavals = linspace(0,30,30)/n;
lossvals = zeros(length(lambdavals),numvalidsets);

for i = 1:length(lambdavals)
    for k = 1:numvalidsets
        X = Xtrain(cvp.training(k),:);
        y = ytrain(cvp.training(k),:);
        Xvalid = Xtrain(cvp.test(k),:);
        yvalid = ytrain(cvp.test(k),:);

        nca = fscnca(X,y,'FitMethod','exact', ...
             'Solver','lbfgs','Lambda',lambdavals(i), ...
             'IterationLimit',60,'GradientTolerance',1e-3, ...
             'Standardize',true);
                  
        lossvals(i,k) = loss(nca,Xvalid,yvalid,'LossFunction','classiferror');
    end
end

meanloss = mean(lossvals,2);

figure()
plot(lambdavals,meanloss,'ro-')
xlabel('Lambda')
ylabel('Loss (MSE)')
grid on

[~,idx] = min(meanloss) % Find the index

bestlambda = lambdavals(idx) % Find the best lambda value

bestloss = meanloss(idx)

nca_LumenNuclei_GLCMstats = fscnca(Xtrain,ytrain,'FitMethod','exact','Solver','lbfgs',...
    'Lambda',bestlambda,'Standardize',true,'Verbose',1);

figure()
plot(nca_LumenNuclei_GLCMstats.FeatureWeights,'ro')
xlabel('Feature index')
ylabel('Feature weight')
grid on

% tol    = 0.1;
% selidx = find(nca_Color_Gamma.FeatureWeights > tol*max(1,max(nca_Color_Gamma.FeatureWeights)));
% 
L = loss(nca_LumenNuclei_GLCMstats,Xtest,ytest)

save('nca_LumenNuclei_GLCMstats.mat','nca_LumenNuclei_GLCMstats')

% features = Xtrain(:,selidx);
% 
% classNames = {'G0','G3','G4','G5'};
% t = templateSVM('Standardize',true,'KernelFunction','linear',"Type","classification","Solver",'ISDA');
% svmMdl = fitcecoc(features,ytrain,'ClassNames',classNames,'Prior','uniform','Learners',t,'Coding','onevsone');
% 
% YPred = predict(svmMdl, Xtest(:,selidx));
% 
% L = loss(svmMdl,Xtest(:,selidx),ytest)
% 
% C = confusionmat(ytest, YPred);
% 
% % Calcular las métricas de desempeño
% accuracy = sum(diag(C)) / sum(C(:));
% 
% title = "Clasificación Multiclase SVM "+ string(accuracy);
% 
% YPred = string(YPred);
% 
% figure
% 
% cm = confusionchart(ytest,YPred, ...
%     'ColumnSummary','column-normalized', ...
%     'RowSummary','row-normalized',...
%     'Title',title);

