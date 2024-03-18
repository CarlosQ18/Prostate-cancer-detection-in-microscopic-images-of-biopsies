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
      ID_Image_name_contents = dir(ID_Image_name_path);

      patch_folder = fullfile(ID_Image_name_path, 'Parches');
      original_folder = fullfile(patch_folder, 'Original');

      if ~exist(original_folder, 'dir')
          mkdir(patch_folder);
          mkdir(original_folder);
          tissue_names = {ID_Image_name_contents.name};
          tissue_names = tissue_names(~ismember(tissue_names, {'.', '..'}));
          Rgbtissue_name = tissue_names{1};

          rgbtissue = imread(fullfile(ID_Image_name_path, Rgbtissue_name));
          Patch_size = 512;

          A_3_ImagePatchingOverlapping(rgbtissue,ID_Image_name,original_folder,Patch_size);
      end

   end
end


%     tr = Tiff(fullfile(name_path, file_name),'r');
%     rgbImage = read(tr);

%                       write_tiff_file = Tiff(patch_path,'w');
%                       tagstruct.ImageLength = patch_size;
%                       tagstruct.ImageWidth =  patch_size;
%                       tagstruct.Photometric = getTag(tr,'Photometric');
%                       tagstruct.BitsPerSample = getTag(tr,'BitsPerSample');
%                       tagstruct.SamplesPerPixel = getTag(tr,'SamplesPerPixel');
%                       tagstruct.PlanarConfiguration = getTag(tr,'PlanarConfiguration');  
%                       setTag(write_tiff_file,tagstruct); 
%                       write(write_tiff_file,rgbBlock);