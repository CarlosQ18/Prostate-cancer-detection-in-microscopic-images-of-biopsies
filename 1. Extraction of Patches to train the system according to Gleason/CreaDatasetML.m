main_folder = pwd;

Datasets_path = fullfile(pwd,'PatchesInfoDatasets3.txt');

if ~exist(Datasets_path)
    error('El archivo PatchesInfoDatasets3.txt no existe ');
    return;
else
    Datasets_table = readtable(Datasets_path);
end

for i = 1: height(Datasets_table)
    id = Datasets_table.ID(i);
    name = string(Datasets_table.Patch(i));
    Patch_name = name + '.png' ;
    Patch_name_path = fullfile(pwd,Datasets_table.Direccion(i), Patch_name);
    Majority = Datasets_table.Majority_Gleason_Pattern(i);
    Minority = Datasets_table.Minority_Gleason_Pattern(i);
    if (Majority == Minority)
        New_directory = 'C:\Users\Carlos F. Quintero\Desktop\Data';
        Name_copy= string(id) + '_' + name+ '.png';
        New_name_path = fullfile(New_directory,string(Majority),Name_copy);
        copyfile(Patch_name_path,New_name_path);
    end
end

