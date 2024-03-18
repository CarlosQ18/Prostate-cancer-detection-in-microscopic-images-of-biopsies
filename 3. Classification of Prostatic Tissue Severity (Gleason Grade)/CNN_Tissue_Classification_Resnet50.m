
%A_1_CrearParches;

clear all;
clc;

% Import TensorFlow and load the model
tf = py.importlib.import_module('tensorflow');
np = py.importlib.import_module('numpy');
model = py.importlib.import_module('tensorflow.keras.models');

main_folder = pwd;

Modelo = model.load_model('C:\Users\Carlos F. Quintero\Desktop\Test Definitivo\Resnet50\Resnet50');

PathTissueClassification = string(pwd) + '\'+'Tissue_Classification_Resnet50.txt'; 
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
      if ~ismember(string(ID_Image_name),string(Tissue.Tisue_ID))
          ID_Image_name_path = fullfile(Grado_folder_path, ID_Image_name);
    
          patch_folder = fullfile(ID_Image_name_path, 'Parches','Original');
          patch_folder_python = strrep(patch_folder, '\', '/');
          decoded_labels = py.Predicted_model_script.Predicted_model(pyargs('ruta_directorio', patch_folder_python, 'model', Modelo));
    
          % Convert the result to a MATLAB cell array
          decoded_labels_matlab = string(decoded_labels);
           
          numeros = str2double(decoded_labels_matlab);
            
          % Crear una estructura para almacenar los resultados
          S = struct('Tisue_ID',ID_Image_name,'Num_Grade_0', 0, 'Num_Grade_3', 0, 'Num_Grade_4', 0, 'Num_Grade_5', 0, ...
                       'Percent_Grade_0', 0, 'Percent_Grade_3', 0, 'Percent_Grade_4', 0, 'Percent_Grade_5', 0, 'True_Tissue_Grade',i);
            
          % Contar las ocurrencias
          conteos = histcounts(numeros, [0, 3, 4, 5, inf]);
            
          % Asignar los resultados a la estructura
          S.Num_Grade_0 = conteos(1);
          S.Num_Grade_3 = conteos(2);
          S.Num_Grade_4 = conteos(3);
          S.Num_Grade_5 = conteos(4);
            
          % Calcular los porcentajes
          total = length(numeros);
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