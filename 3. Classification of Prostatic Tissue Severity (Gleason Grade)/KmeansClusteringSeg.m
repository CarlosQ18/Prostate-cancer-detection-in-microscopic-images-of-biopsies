function [Lumen,Nuclei] = KmeansClusteringSeg (he)
    numColors = 3;
    L = imsegkmeans(he,numColors);
    B = labeloverlay(he,L);
    % figure
    % imshow(B);
    % title("Labeled Image RGB");

    lab_he = rgb2lab(he);

    ab = lab_he(:,:,2:3);
    ab = im2single(ab);
    pixel_labels = imsegkmeans(ab,numColors,NumAttempts=10);

    B2 = labeloverlay(he,pixel_labels);
    % figure
    % imshow(B2);
    % title("Labeled Image a*b*");

    mask1 = pixel_labels == 1;
    cluster1 = he.*uint8(mask1);
    % figure
    % imshow(cluster1)
    % title("Objects in Cluster 1");

    mask2 = pixel_labels == 2;
    cluster2 = he.*uint8(mask2);
    % figure
    % imshow(cluster2)
    % title("Objects in Cluster 2");

    mask3 = pixel_labels == 3;
    cluster3 = he.*uint8(mask3);
    % figure
    % imshow(cluster3);
    % title("Objects in Cluster 3");

    % mask4 = pixel_labels == 4;
    % cluster4 = he.*uint8(mask4);
    % figure
    % imshow(cluster4);
    % title("Objects in Cluster 4");


    Hematoxilina_Eosina = {cluster1,cluster2,cluster3};
    Hematoxilina_Eosina_masks = {mask1,mask2,mask3};

    Conteo = [];

    for i = 1:3
        cluster = Hematoxilina_Eosina{i};
        isLumen = any(cluster(:,:,2) >= 244 & cluster(:,:,2) <= 255);
        Num_isLumen = sum( isLumen == 1,'all');
        Conteo(end+1) = Num_isLumen;
    end

    [~,indice] = max(Conteo);
    Lumen = Hematoxilina_Eosina{indice};
    % figure
    % imshow(Lumen)
    % title("Lumen")

    Hematoxilina_Eosina(indice) = [];
    Hematoxilina_Eosina_masks(indice) = [];

    A = any(Hematoxilina_Eosina{1}(:,:,2) > 30 & Hematoxilina_Eosina{1}(:,:,2) < 40);
    NUM_A = sum( A == 1,'all');

    B = any(Hematoxilina_Eosina{2}(:,:,2) > 30 & Hematoxilina_Eosina{2}(:,:,2) < 40);
    NUM_B = sum( B == 1,'all');

    if NUM_A > NUM_B
        Hematoxilina = Hematoxilina_Eosina{1};
        Mask_Hema = Hematoxilina_Eosina_masks{1};
        % figure
        % imshow(Hematoxilina)
        % title("Hematoxilina")
        Eosina = Hematoxilina_Eosina{2};
        % figure
        % imshow(Eosina)
        % title("Eosina")
    else
        Hematoxilina = Hematoxilina_Eosina{2};
        Mask_Hema = Hematoxilina_Eosina_masks{2};
        % figure
        % imshow(Hematoxilina)
        % title("Hematoxilina")
        Eosina = Hematoxilina_Eosina{1};
        % figure
        % imshow(Eosina)
        % title("Eosina")
    end


    L = lab_he(:,:,1);
    L_blue = L.*double(Mask_Hema);
    L_blue = rescale(L_blue);
    idx_light_blue = imbinarize(nonzeros(L_blue));

    blue_idx = find(Mask_Hema);
    mask_dark_blue = Mask_Hema;
    mask_dark_blue(blue_idx(idx_light_blue)) = 0;
    
    Nuclei = he.*uint8(mask_dark_blue);
    % figure
    % imshow(Nuclei)
    % title("Blue Nuclei")

end