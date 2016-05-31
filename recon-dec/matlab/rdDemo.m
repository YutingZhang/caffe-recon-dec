function rdDemo

% add neccessary paths
RDMAT_ROOT = fileparts( mfilename('fullpath') );
addpath(RDMAT_ROOT);
rdInit;

% gpu mode
caffe.set_mode_gpu();

% input image
input_img_path = fullfile( RDMAT_ROOT, 'ILSVRC2012_val_00000011.JPEG' );
I = imread(input_img_path);


% reconstruction (VGGNet, layer5)

sae_first   = rdNet('vggnet/recon/SAE-first/layer5');
swwae_first = rdNet('vggnet/recon/SWWAE-first/layer5');
[R_sae_first, ~, I_resized]   = rdRun(sae_first,I);
R_swwae_first = rdRun(swwae_first,I);

figure;
clf
subplot(1,3,1);
imshow( I_resized );
title( 'input' );
subplot(1,3,2);
imshow( R_sae_first );
title( sprintf( '%s-based\nSAE-first\n(5th layer)', 'vggnet') );
subplot(1,3,3);
imshow( R_swwae_first );
title( sprintf( '%s-based\nSWWAE-first\n(5th layer)', 'vggnet') );

% reconstruction (AlexNet, layer5)

swwae_first = rdNet('alexnet/recon/SWWAE-first/layer5');
[R_swwae_first, ~, I_resized]   = rdRun(swwae_first,I);

figure;
clf
subplot(1,2,1);
imshow( I_resized );
title( 'input' );
subplot(1,2,2);
imshow( R_swwae_first );
title( sprintf( '%s-based\nSWWAE-first\n(5th layer)', 'alexnet') );


% cls + reconstruction
%swwae_all   = rdNet('vggnet/cls/SWWAE-all/layer5');
%sae_all     = rdNet('vggnet/cls/SAE-all/layer5');

