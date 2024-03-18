LumenNuclei = readtable(fullfile(pwd,'LumenNuclei_GLCMstats.txt'));
Color = readtable(fullfile(pwd,'Colorstats_Data 2.txt'));


Image = LumenNuclei(:,1);
Grado = LumenNuclei(:,end);

LumenNuclei = LumenNuclei(:,2:end-1);
Color = Color(:,2:end-1);


% Concatenar las tres tablas
result_table = [Image, Color,LumenNuclei,Grado];

Name = strcat('Color_GLCMLumenNuclei','.txt');
Stats_path = fullfile(pwd,Name);

if ~exist(Stats_path,'file')
    writetable(result_table, Stats_path, "WriteRowNames",true);
end