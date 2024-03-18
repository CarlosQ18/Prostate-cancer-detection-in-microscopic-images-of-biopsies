Clases = 4;
Path = 'C:\Users\Carlos F. Quintero\Desktop\Dataset';
Tipo = "Train";
folder_path = fullfile(Path, Tipo);
Image_contents = dir(folder_path);
Score_folder_names = {Image_contents.name};
Score_folder_names = Score_folder_names(~ismember(Score_folder_names, {'.', '..'}));

Name = 'Wavelet_GLCMstats_'+ Tipo +'.txt';
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
            % Aplicar transformada wavelet de nivel 1
            [LL1,LH1,HL1,HH1] = dwt2(I,'haar');

            % Aplicar transformada wavelet de nivel 2
            [LL2,LH2,HL2,HH2] = dwt2(LL1,'haar');

            Cuadrantes = {LH1,HL1,HH1,LH2,HL2,HH2};
            Nombres = ["LH1","HL1","HH1","LH2","HL2","HH2"];

            ID_Image = table({string(Image_ID)},'VariableNames',{'Image'});
            Grado = table([str2num(Score_folder_names{i})],'VariableNames',{'Grado'});

            AuxiliarTable = table;
            for k = 1:6

                Cuadrante = Cuadrantes{1,k};
                Nombre = Nombres(k);


                GLCM2 = graycomatrix(Cuadrante,'Offset',[2 0;0 2]);
                stats = GLCM_Features1(GLCM2,1);

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
    

                % Crear los nombres de las columnas
                nombresColumnas = [string(Nombre) + "_Autoc", string(Nombre) + "_Contr", string(Nombre) + "_Corrm", string(Nombre) + "_Corrp", ...
                                   string(Nombre) + "_Cprom", string(Nombre) + "_Cshad", string(Nombre) + "_Dissi", string(Nombre) + "_Energ", ...
                                   string(Nombre) + "_Entro", string(Nombre) + "_Homom", string(Nombre) + "_Homop", string(Nombre) + "_Maxpr", ...
                                   string(Nombre) + "_Sosvh", string(Nombre) + "_Savgh", string(Nombre) + "_Svarh", string(Nombre) + "_Senth",...
                                   string(Nombre) + "_Dvarh", string(Nombre) + "_Denth", string(Nombre) + "_Inf1h", string(Nombre) + "_Inf2h",...
                                   string(Nombre) + "_Indnc", string(Nombre) + "_Idmnc"];
                
                % Crear la tabla con los nombres de las columnas
                tabla = table([Autoc],[Contr],[Corrm],[Corrp],[Cprom],[Cshad],[Dissi],[Energ],[Entro],[Homom],[Homop],[Maxpr],[Sosvh],[Savgh],[Svarh],[Senth],[Dvarh],[Denth],[Inf1h],[Inf2h],[Indnc],[Idmnc], 'VariableNames', convertStringsToChars(nombresColumnas));


                
                AuxiliarTable = [AuxiliarTable,tabla];
            end
            tabla = [ID_Image,AuxiliarTable,Grado];

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