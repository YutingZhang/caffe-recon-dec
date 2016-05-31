function rdDemo

% add neccessary paths
RDMAT_ROOT = fileparts( mfilename('fullpath') );
addpath(RDMAT_ROOT);
rdInit;

% input image
input_img_path = fullfile( RDMAT_ROOT, 'ILSVRC2012_val_00000011.JPEG' );
I = imread(input_img_path);

% reconstruction
sae_first   = rdNet('vggnet/recon/SAE-first/layer5');
swwae_first = rdNet('vggnet/recon/SWWAE-first/layer5');
R_sae_first   = rdRun(sae_first,I);
R_swwae_first = rdRun(swwae_first,I);

figure(1);
clf
subplot(1,3,1);
imshow( I );
title( 'input' );
subplot(1,3,2);
imshow( R_sae_first );
title( 'SAE-first (5th layer)' );
subplot(1,3,3);
imshow( R_swwae_first );
title( 'SWWAE-first (5th layer)' );


% cls + reconstruction
swwae_all   = rdNet('vggnet/cls/SWWAE-all/layer5');
sae_all     = rdNet('vggnet/cls/SAE-all/layer5');

