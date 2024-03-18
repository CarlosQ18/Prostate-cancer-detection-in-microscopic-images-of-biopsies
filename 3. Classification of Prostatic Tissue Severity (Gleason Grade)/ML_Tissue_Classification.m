clear all;
clc;

A_1_CrearParches;

clear all;
clc;

main_folder = pwd;

classifier =  py.joblib.load("D:\Test Definitivo\SVMModel__74_Features_RF_75 (9Fold).joblib");

Features =  int64(py.joblib.load("C:\Users\Carlos F. Quintero\Desktop\Test Definitivo\RF_Selected_Indices_74_features_py.joblib"));
Features =  Features + 1; %Se suma +1 debido a que python comienza en 0

PathTissueClassification = string(pwd) + '\'+'Tissue_Classification_SVM_9Fold_Overlapping.txt'; 
if exist(PathTissueClassification,'file')
  Tissue = readtable(PathTissueClassification);
else
  nombresColumnas = {'Tisue_ID','Num_Grade_0', 'Num_Grade_3', 'Num_Grade_4','Num_Grade_5','Percent_Grade_0', 'Percent_Grade_3', 'Percent_Grade_4', 'Percent_Grade_5', 'True_Tissue_Grade'};
  Tissue = cell2table(cell(0, length(nombresColumnas)), 'VariableNames', nombresColumnas);
end

for i = 0:5
   Grado_folder = sprintf('Grado %d', i);
   Grado_folder_path = fullfile(main_folder, Grado_folder);
   Grado_folder_contents = dir(Grado_folder_path);
   
   Grado_folder_names = {Grado_folder_contents.name};
   Grado_folder_names = Grado_folder_names(~ismember(Grado_folder_names, {'.', '..'}));
   
   for j = 1:numel(Grado_folder_names)
      ID_Image_name = Grado_folder_names{j};
      ID_Image_name_path = fullfile(Grado_folder_path, ID_Image_name);
      Name = strcat(ID_Image_name,'_Color_Lumen_Nuclei','.txt');
      Stats_path = fullfile(ID_Image_name_path,Name);

      if ~ismember(string(ID_Image_name),string(Tissue.Tisue_ID))

          if ~exist(Stats_path,'file')
              Color_Lumen_Nuclei_Table = CreateColorLumenTable(ID_Image_name_path,ID_Image_name);
          else
              Color_Lumen_Nuclei_Table = readtable(Stats_path);
          end
          
          Color_Lumen_Nuclei_Table = table2array(Color_Lumen_Nuclei_Table);
          Color_Lumen_Nuclei_Table = Color_Lumen_Nuclei_Table(:,Features);
          filas_con_nan = any(isnan(Color_Lumen_Nuclei_Table), 2);
          Color_Lumen_Nuclei_Table = Color_Lumen_Nuclei_Table(~filas_con_nan, :);

          y_pred = double(classifier.predict(Color_Lumen_Nuclei_Table));

                    % Crear una estructura para almacenar los resultados
          S = struct('Tisue_ID',ID_Image_name,'Num_Grade_0', 0, 'Num_Grade_3', 0, 'Num_Grade_4', 0, 'Num_Grade_5', 0, ...
                       'Percent_Grade_0', 0, 'Percent_Grade_3', 0, 'Percent_Grade_4', 0, 'Percent_Grade_5', 0, 'True_Tissue_Grade',i);
            
          % Contar las ocurrencias
          conteos = histcounts(y_pred, [0, 3, 4, 5, inf]);
            
          % Asignar los resultados a la estructura
          S.Num_Grade_0 = conteos(1);
          S.Num_Grade_3 = conteos(2);
          S.Num_Grade_4 = conteos(3);
          S.Num_Grade_5 = conteos(4);
            
          % Calcular los porcentajes
          total = length(y_pred);
          S.Percent_Grade_0 = (S.Num_Grade_0 / total) * 100;
          S.Percent_Grade_3 = (S.Num_Grade_3 / total) * 100;
          S.Percent_Grade_4 = (S.Num_Grade_4 / total) * 100;
          S.Percent_Grade_5 = (S.Num_Grade_5 / total) * 100;
    
          Table = struct2table(S);
    
          Tissue = [Tissue;Table];

      else
          continue;
      end
   end
end

Tissue = sortrows(Tissue, {'True_Tissue_Grade', 'Tisue_ID'});
writetable(Tissue, PathTissueClassification, "WriteRowNames",true);