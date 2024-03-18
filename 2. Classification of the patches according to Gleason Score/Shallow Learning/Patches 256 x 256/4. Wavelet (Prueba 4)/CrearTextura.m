Clases = 4;
Path = 'C:\Users\Carlos F. Quintero\Desktop\Dataset';
Tipo = "Train";
folder_path = fullfile(Path, Tipo);
Image_contents = dir(folder_path);
Score_folder_names = {Image_contents.name};
Score_folder_names = Score_folder_names(~ismember(Score_folder_names, {'.', '..'}));

Name = 'GLCMstats_'+ Tipo +'.txt';
Stats_path = fullfile(pwd,Name);
if ~exist(Stats_path,'file')
    Stats_table = table;
    Stats_table.Image = {};
else
    Stats_table = readtable(Stats_path);
end

for i = 1:Clases
    Score_Folder_name = fullfile(folder_path,Score_folder_names{i});
    Image_contents = dir(Score_Folder_name);
    Image_names = {Image_contents.name};
    Image_names = Image_names(~ismember(Image_names, {'.', '..'}));
    for j = 1:numel(Image_names)
        Image_ID = Image_names(j);
        Indice = find(string(Stats_table.Image) == string(Image_ID));
        if isempty(Indice)
            image_path = fullfile(Score_Folder_name,Image_ID);
            I = imread(image_path);
            I = rgb2gray(I);
            GLCM2 = graycomatrix(I,'Offset',[2 0;0 2]);
            stats = GLCM_Features1(GLCM2,1);
            Image = string(Image_ID);
            Autoc = stats.autoc;
            Contr = stats.contr;
            Corrm = stats.corrm;
            Corrp = stats.corrp;
            Cprom = stats.cprom;
            Cshad = stats.cshad;
            Dissi = stats.dissi;
            Energ = stats.energ;
            Entro = stats.entro;
            Homom = stats.homom;
            Homop = stats.homop;
            Maxpr = stats.maxpr;
            Sosvh = stats.sosvh;
            Savgh = stats.savgh;
            Svarh = stats.svarh;
            Senth = stats.senth;
            Dvarh = stats.dvarh;
            Denth = stats.denth;
            Inf1h = stats.inf1h;
            Inf2h = stats.inf2h;
            Indnc = stats.indnc;
            Idmnc = stats.idmnc;
            Grado = str2num(Score_folder_names{i});
            tabla = table({string(Image)},[Autoc],[Contr],[Corrm],[Corrp],[Cprom],[Cshad],[Dissi],[Energ],[Entro],[Homom],[Homop],[Maxpr],[Sosvh],[Savgh],[Svarh],[Senth],[Dvarh],[Denth],[Inf1h],[Inf2h],[Indnc],[Idmnc],[Grado],...
                                           'VariableNames',{'Image','Autoc','Contr','Corrm','Corrp','Cprom','Cshad','Dissi','Energ','Entro','Homom','Homop','Maxpr','Sosvh','Savgh','Svarh','Senth','Dvarh','Denth','Inf1h','Inf2h','Indnc','Idmnc','Grado'});
            if ~exist(Stats_path,'file')
                writetable(tabla, Stats_path, "WriteRowNames",true);
            else
                writetable(tabla,Stats_path,'WriteMode','Append','WriteVariableNames',false,'WriteRowNames',true);
            end
        else
            continue;
        end
    end
end