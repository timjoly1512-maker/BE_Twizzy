close all
clear all
clc

I = imread("C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\valid\00004_00048_00027_png.rf.31be7c974beed30e90aaba9f74f293f0.jpg");
figure, imshow(I);

ihsv = rgb2hsv(I);
Ih = ihsv(:,:,1);

figure
imhist(Ih);

masqueR = zeros(size(Ih));
masqueR(Ih>0.95) = 1;

masqueJ = zeros(size(Ih));
masqueJ(Ih<0.18 & Ih>0.05) = 1;

masque = masqueR | masqueJ;

figure
imshow(masque);

se = strel("disk",5);
masqueerode = imerode(masque,se);
masquedilate = imdilate(masqueerode, se);

figure
imshowpair(masque, masqueerode,"montage");

figure
imshowpair(masqueerode,masquedilate, "montage");

% Détection de contours
figure
contourim = edge(I(:,:,1));
imshow(contourim);

figure
contourim2 = edge(masquedilate);
imshow(contourim2);

% Détection de cercles
figure
[centers, radii] = imfindcircles(contourim, [90 500], Sensitivity=0.98);
imshow(I);
viscircles(centers,radii, Color='m');

I1 = imread("C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\valid\00004_00048_00027_png.rf.31be7c974beed30e90aaba9f74f293f0.jpg");
I2 = imread("C:\twizy\BDD_Roboflow_Tazi\BDD_Roboflow_Tazi\valid\00004_00040_00022_png.rf.0d800d480d982db614422c14964eaf28.jpg");

[rows, cols, ~] = size(I1);
[cols_grid, rows_grid] = meshgrid(1:cols, 1:rows);

center = centers(1,:);
radius = radii(1);

distance = sqrt((cols_grid - center(1)).^2 + (rows_grid - center(2)).^2);
mask = distance <= radius;

% Adapter le masque en 3D
mask3 = repmat(mask, [1 1 3]);

I1_gray = rgb2gray(I1);
I2_gray = rgb2gray(I2);

% garder uniquement les pixels du masque
I11 = I1_gray;
I22 = I2_gray;

% Détection de points SURF
points1 = detectSURFFeatures(I11);
points2 = detectSURFFeatures(I22);

[f1,vpts1] = extractFeatures(I11,points1);
[f2,vpts2] = extractFeatures(I22,points2);

indexPairs = matchFeatures(f1,f2,MaxRatio=0.9);

matchedPoints1 = vpts1(indexPairs(:,1),:);
matchedPoints2 = vpts2(indexPairs(:,2),:);

figure
showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2, "montage", 'PlotOptions', {'ro','go','y-'});
title("Candidats");