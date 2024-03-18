
PathTrain = "D:\Dataset\Train";

imData = imageDatastore(PathTrain, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

Grado = imData.Labels;

direcciones = imData.Files;
[~, Image, Extension] = cellfun(@fileparts, direcciones, 'UniformOutput', false);
Image = strcat(Image,Extension);

Image = cell2table(Image);
Grado = array2table(Grado);

Dataset = readall(imData);

LumenTable = table;
NucleiTable = table;

Tablas = {LumenTable,NucleiTable};

for i = 1:length(Dataset)
    disp(i);
    he = Dataset{i,1};
    [Lumen,Nuclei] = KmeansClusteringSeg(he);
    Lumen_Nuclei = {Lumen,Nuclei};
    Gray_HSV_Table = table;
    for j = 1:length(Lumen_Nuclei)
        img = Lumen_Nuclei{j};
        gray = rgb2gray(img);
        GLCM = graycomatrix(gray,'Offset',[0 1;-1 1;-1 0;-1 -1],'Symmetric',true);
        GLCM = sum(GLCM,3);
        stats = GLCM_Features1(GLCM,0);
        statsgray = struct2table(stats);
        HSV = rgb2hsv(img);
        Channels = ['H','S','V'];
        HSV_Table = table;
        for k = 1:length(Channels)
            HSVChannel = HSV(:,:,k);
            GLCM = graycomatrix(HSVChannel,'Offset',[0 1;-1 1;-1 0;-1 -1],'Symmetric',true);
            GLCM = sum(GLCM,3);
            stats = GLCM_Features1(GLCM,0);
            statsHSV = struct2table(stats);
            if k == 1
                statsHSV.Properties.VariableNames = strcat('H_', statsHSV.Properties.VariableNames);
            elseif k == 2
                statsHSV.Properties.VariableNames = strcat('S_', statsHSV.Properties.VariableNames);
            else
                statsHSV.Properties.VariableNames = strcat('V_', statsHSV.Properties.VariableNames);
            end
            HSV_Table = [HSV_Table,statsHSV];
        end
        Gray_HSV_Table = [statsgray, HSV_Table];
        Tablas{1,j} = [Tablas{1,j};Gray_HSV_Table];
    end
end

LumenTable = Tablas{1,1};
LumenTable.Properties.VariableNames = strcat('Lumen_', LumenTable.Properties.VariableNames);

NucleiTable = Tablas{1,2};
NucleiTable.Properties.VariableNames = strcat('Nuclei_', NucleiTable.Properties.VariableNames);

Lumen_Nuclei_Table = [Image,LumenTable,NucleiTable,Grado];

Name = strcat('LumenNuclei_GLCMstats','.txt');
Stats_path = fullfile(pwd,Name);

if ~exist(Stats_path,'file')
    writetable(Lumen_Nuclei_Table, Stats_path, "WriteRowNames",true);
end
