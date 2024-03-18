function A_2_ImagePatching (rgbImage,masktissue, ID_image,save_folder, patch_size)     
      [rows,columns,numberOfColorBands] = size(rgbImage);
      
      % The first way to divide an image up into blocks is by using mat2cell().
      blockSizeR = patch_size; % Rows in block.
      blockSizeC = patch_size; % Columns in block.
        
      % Figure out the size of each block in rows. Most will be blockSizeR but there may be a remainder amount of less than that.
      wholeBlockRows = floor(rows / blockSizeR);
      blockVectorR = [blockSizeR * ones(1, wholeBlockRows), rem(rows, blockSizeR)];
      % Figure out the size of each block in columns.
      wholeBlockCols = floor(columns / blockSizeC);
      blockVectorC = [blockSizeC * ones(1, wholeBlockCols), rem(columns, blockSizeC)];
       
      % Create the cell array, ca. Each cell (except for the remainder cells at the end of the image)
      % in the array contains a blockSizeR by blockSizeC by 3 color array.
      % This line is where the image is actually divided up into blocks.
      if numberOfColorBands > 1
          % It's a color image.
          ca = mat2cell(rgbImage, blockVectorR, blockVectorC, numberOfColorBands);
      else
          ca = mat2cell(rgbImage, blockVectorR, blockVectorC);
      end       
      % Save all the blocks.
      numPlotsR = size(ca, 1);
      numPlotsC = size(ca, 2);
      for r = 1 : numPlotsR
          for c = 1 : numPlotsC
              Filename = string(r)+ "_" + string(c) + ".png";
              patch_path = fullfile(save_folder,Filename);
              if ~isfile(patch_path)
                  % Extract the numerical array out of the cell
                  subImage = ca{r,c};
                  MeanGeneral=mean2(subImage);
                  StdGeneral=std2(subImage);

                  [Rojo_pixelCounts,Rojo_GLs] = imhist(subImage(:,:,1));
                  [MeanRed,StdRed,~,~,~] = GetMoments(Rojo_GLs, Rojo_pixelCounts);
        
                  [Verde_pixelCounts,Verde_GLs] = imhist(subImage(:,:,2));
                  [MeanGreen,StdGreen,~,~,~] = GetMoments(Verde_GLs, Verde_pixelCounts);
        
                  [Azul_pixelCounts,Azul_GLs] = imhist(subImage(:,:,3));
                  [MeanBlue,StdBlue,~,~,~] = GetMoments(Azul_GLs, Azul_pixelCounts);

                  Filtro_Medias = (MeanGeneral > 150 && MeanGeneral < 198) && (MeanRed > 167 && MeanRed < 239) && (MeanGreen > 115 && MeanGreen < 185) && (MeanBlue > 144 && MeanBlue < 200);
                  Filtro_Std = (StdGeneral > 22 && StdGeneral < 57) && (StdRed > 12 && StdRed < 54) && (StdGreen > 23 && StdGreen < 64) && (StdBlue > 16 && StdBlue < 49);
                
                  if ( Filtro_Medias && Filtro_Std )
                      [Size_row, Size_col, numChannel] = size(subImage);
                      fila_inicial = (r - 1) * patch_size + 1;
                      col_inicial =  (c - 1) * patch_size + 1;
                      %fprintf('c=%d, r=%d fila_inicial=%d col_inicial=%d size=%d %d\n', c, r, fila_inicial, col_inicial,Size_col,Size_row);
                      patch_mask = masktissue(fila_inicial:(fila_inicial+Size_row)-1, col_inicial:(col_inicial+Size_col)-1);
                      labels = patch_mask( patch_mask >= 2);
                      if ~isempty(labels)
                          GleasonPattern = unique(labels); 
                          Frecuencias = histcounts(labels, GleasonPattern); 
                          [~, i1] = min(Frecuencias);
                          MajorityGleasonPattern = mode(labels);
                          if (MajorityGleasonPattern == 2)
                              MajorityGleasonPattern = 0;
                          end
                          MinorityGleasonPattern = GleasonPattern(i1);
                          if (MinorityGleasonPattern == 2)
                              MinorityGleasonPattern = 0;
                          end
                          if (MajorityGleasonPattern == MinorityGleasonPattern)
                              startIndex = strfind(save_folder, 'Datasets') + length('Datasets') + 1;
                              shortPath = save_folder(startIndex:end);
                              tabla = table({string(ID_image)},{string(r)+'_'+string(c)},[MajorityGleasonPattern],[MinorityGleasonPattern],{shortPath},...
                                           'VariableNames',{'ID','Patch','Majority_Gleason_Pattern','Minority_Gleason_Pattern','Direccion'});
                              PathSaveInfo = string(pwd) + '\'+'PatchesInfoDatasets3.txt'; 
                              if ~exist(PathSaveInfo,'file')
                                  writetable(tabla, PathSaveInfo, "WriteRowNames",true);
                              else
                                  writetable(tabla,PathSaveInfo,'WriteMode','Append','WriteVariableNames',false,'WriteRowNames',true);
                              end
                              imwrite(subImage,patch_path,"png");
                          end
                      end
                  end
              end
           end
      end
end


%  rgbImage = imread("C:\Users\Carlos F. Quintero\Desktop\Datasets\Grado 0\06a0cbd8fd6320ef1aa6f19342af2e68\06a0cbd8fd6320ef1aa6f19342af2e68.tiff");
%        masktissue = imread("C:\Users\Carlos F. Quintero\Desktop\Datasets\Grado 0\06a0cbd8fd6320ef1aa6f19342af2e68\06a0cbd8fd6320ef1aa6f19342af2e68_mask.tiff");
%        masktissue = masktissue(:,:,1);
%       ID_image = '06a0cbd8fd6320ef1aa6f19342af2e68';
%       save_folder = 'C:\Users\Carlos F. Quintero\Desktop\Datasets\Grado 0\06a0cbd8fd6320ef1aa6f19342af2e68\Parches';
%       patch_size = 256;

%                    mean=mean2(rgbBlock);
%                    ds=std2(rgbBlock);
%                    if mean < 230 && ds > 15
% %                       imwrite(rgbBlock,patch_path);
% %                   end
