clear all; close all; clc;


sourceDir = "O:\BE_Twizy\BDD_Roboflow_Tazi\test";
destGood  = "O:\BE_Twizy\BDD_Roboflow_Tazi\valid";
destBad   = "O:\BE_Twizy\BDD_Roboflow_Tazi\PasValid";

if ~exist(destGood, 'dir'), mkdir(destGood); end
if ~exist(destBad, 'dir'), mkdir(destBad); end

s
seuilGmag = 10.0;   
seuilLap  = 50.0;    
lumiMin   = 45;        
lumiMax   = 220;       

fileList = [dir(fullfile(sourceDir, '*.jpg')); dir(fullfile(sourceDir, '*.png'))];
fprintf('Traitement de %d images en cours...\n', length(fileList));

for i = 1:length(fileList)
    imgName = fileList(i).name;
    fullPath = fullfile(sourceDir, imgName);
    
    try
        I = imread(fullPath);
        if size(I, 3) == 3, Igray = rgb2gray(I); else, Igray = I; end
        

        lapFilter = [0 1 0; 1 -4 1; 0 1 0];
        Ilap = imfilter(double(Igray), lapFilter, 'replicate');
        scoreFlou = var(Ilap(:));
        
        [~, Gmag] = imgradient(Igray);
        scoreContraste = mean(Gmag(:)); 
        
        scoreLumi = mean(Igray(:));

        if (scoreFlou > seuilLap) && (scoreContraste > seuilGmag) && ...
           (scoreLumi > lumiMin) && (scoreLumi < lumiMax)
            copyfile(fullPath, fullfile(destGood, imgName));
        else
            copyfile(fullPath, fullfile(destBad, imgName));
        end
        
    catch
        fprintf('Erreur sur : %s\n', imgName);
    end
end
fprintf('Termine ! Score netteté moyen utilisé : %f\n', seuilLap);