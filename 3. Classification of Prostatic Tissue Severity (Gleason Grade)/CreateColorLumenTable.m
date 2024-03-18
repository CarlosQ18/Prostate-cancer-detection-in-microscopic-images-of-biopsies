function [Color_Lumen_Nuclei_Table] = CreateColorLumenTable(Directory,ID_Patch)
    Path = fullfile(Directory,'Parches','Original');
    imds = imageDatastore(Path);
    Dataset = readall(imds);

    LumenTable = table;
    NucleiTable = table;
    
    Tablas = {LumenTable,NucleiTable};

    RGB_tabla = table;

    for i = 1:length(Dataset)
        he = Dataset{i};

        Colores = {he(:,:,1),he(:,:,2),he(:,:,3)};
        Nombres = ["R","G","B"];

        RGB_tabla_Element = table;

        for j = 1:3

            Color = Colores{1,j};
            Nombre = Nombres(j);
            [pixelCounts,GLs] = imhist(Color);
            [meanGL,sd,varianceGL,skew,kurtosis] = GetMoments(GLs, pixelCounts);


            % Crear los nombres de las columnas
            nombresColumnas = [string(Nombre) + "_Mean", string(Nombre) + "_Std", string(Nombre) + "_Varian", string(Nombre) + "_Skew", ...
                               string(Nombre) + "_Kurt"];
                
            % Crear la tabla con los nombres de las columnas
            tabla = table([meanGL],[sd],[varianceGL],[skew],[kurtosis], 'VariableNames', convertStringsToChars(nombresColumnas));

            RGB_tabla_Element = [RGB_tabla_Element, tabla];
        end

        RGB_tabla = [RGB_tabla;RGB_tabla_Element];

        [Lumen,Nuclei] = KmeansClusteringSeg(he);
        Lumen_Nuclei = {Lumen,Nuclei};
        Gray_HSV_Table = table;
        for k = 1:length(Lumen_Nuclei)
            img = Lumen_Nuclei{k};
            gray = rgb2gray(img);
            GLCM = graycomatrix(gray,'Offset',[0 1;-1 1;-1 0;-1 -1],'Symmetric',true);
            GLCM = sum(GLCM,3);
            stats = GLCM_Features1(GLCM,0);
            statsgray = struct2table(stats);
            HSV = rgb2hsv(img);
            Channels = ['H','S','V'];
            HSV_Table = table;
            for l = 1:length(Channels)
                HSVChannel = HSV(:,:,l);
                GLCM = graycomatrix(HSVChannel,'Offset',[0 1;-1 1;-1 0;-1 -1],'Symmetric',true);
                GLCM = sum(GLCM,3);
                stats = GLCM_Features1(GLCM,0);
                statsHSV = struct2table(stats);
                if l == 1
                    statsHSV.Properties.VariableNames = strcat('H_', statsHSV.Properties.VariableNames);
                elseif l == 2
                    statsHSV.Properties.VariableNames = strcat('S_', statsHSV.Properties.VariableNames);
                else
                    statsHSV.Properties.VariableNames = strcat('V_', statsHSV.Properties.VariableNames);
                end
                HSV_Table = [HSV_Table,statsHSV];
            end
            Gray_HSV_Table = [statsgray, HSV_Table];
            Tablas{1,k} = [Tablas{1,k};Gray_HSV_Table];
        end 
    end

    LumenTable = Tablas{1,1};
    LumenTable.Properties.VariableNames = strcat('Lumen_', LumenTable.Properties.VariableNames);
    
    NucleiTable = Tablas{1,2};
    NucleiTable.Properties.VariableNames = strcat('Nuclei_', NucleiTable.Properties.VariableNames);
    
    Color_Lumen_Nuclei_Table = [RGB_tabla,LumenTable,NucleiTable];

    Name = strcat(ID_Patch,'_Color_Lumen_Nuclei','.txt');
    Stats_path = fullfile(Directory,Name);
    
    if ~exist(Stats_path,'file')
        writetable(Color_Lumen_Nuclei_Table, Stats_path, "WriteRowNames",true);
    end

end