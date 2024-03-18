main_folder = pwd;

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

      patch_folder = fullfile(ID_Image_name_path, 'Parches');

      if exist(patch_folder, 'dir')
        rmdir(patch_folder,'s');
        if exist(Stats_path,'file')
            delete(Stats_path);
        end
      end
   end
end