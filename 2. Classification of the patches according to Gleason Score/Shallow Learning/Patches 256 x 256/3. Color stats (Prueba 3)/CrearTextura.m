Clases = 4;
Path = 'C:\Users\Carlos F. Quintero\Desktop\Dataset';
Tipo = "Train";
folder_path = fullfile(Path, Tipo);
Image_contents = dir(folder_path);
Score_folder_names = {Image_contents.name};
Score_folder_names = Score_folder_names(~ismember(Score_folder_names, {'.', '..'}));

Name = 'Colorstats_'+ Tipo +'.txt';
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

            Colores = {I(:,:,1),I(:,:,2),I(:,:,3)};
            Nombres = ["R","G","B"];

            ID_Image = table({string(Image_ID)},'VariableNames',{'Image'});
            Grado = table([str2num(Score_folder_names{i})],'VariableNames',{'Grado'});

            AuxiliarTable = table;
            for k = 1:3

                Color = Colores{1,k};
                Nombre = Nombres(k);
                [pixelCounts,GLs] = imhist(Color);
                [meanGL,sd,varianceGL,skew,kurtosis] = GetMoments(GLs, pixelCounts);
    

                % Crear los nombres de las columnas
                nombresColumnas = [string(Nombre) + "_Mean", string(Nombre) + "_Std", string(Nombre) + "_Varian", string(Nombre) + "_Skew", ...
                                   string(Nombre) + "_Kurt"];
                
                % Crear la tabla con los nombres de las columnas
                tabla = table([meanGL],[sd],[varianceGL],[skew],[kurtosis], 'VariableNames', convertStringsToChars(nombresColumnas));
                
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