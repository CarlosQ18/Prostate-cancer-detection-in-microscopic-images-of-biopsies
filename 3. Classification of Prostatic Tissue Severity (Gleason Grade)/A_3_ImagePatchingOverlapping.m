function A_3_ImagePatchingOverlapping (rgbImage, ID_image,save_folder, patch_size)

    [rows,columns,numColorChannels] = size(rgbImage);
    stepSize = 410;
    subImageWidth = patch_size;

    for row = 1 : stepSize : rows
	    row2 = min(row + subImageWidth - 1, rows);
	    for col = 1 : stepSize : columns
		    col2 = min(col + subImageWidth - 1, columns);
		    subImage = rgbImage(row:row2, col:col2, :);

            Filename = string(ID_image) + "_" + string(row)+ "_" + string(row2) + "_" + string(col) + "_" + string(col2) + ".png";
            patch_path = fullfile(save_folder,Filename);

            if ~isfile(patch_path)

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
                    imwrite(subImage,patch_path,"png");
                end

            end
	    end
    end
end