function rdDemo

% add neccessary paths
RDMAT_ROOT = fileparts( mfilename('fullpath') );
addpath(RDMAT_ROOT);
rdInit;

% gpu mode
caffe.set_gpu_mode();

% input image
input_img_path = fullfile( RDMAT_ROOT, 'ILSVRC2012_val_00000011.JPEG' );
I = imread(input_img_path);

% reconstruction

base_networks = {'alexnet','vggnet'};

for b = 1:2

    sae_first   = rdNet([base_networks{b} '/recon/SAE-first/layer5']);
    swwae_first = rdNet([base_networks{b} '/recon/SWWAE-first/layer5']);
    R_sae_first   = rdRun(sae_first,I);
    R_swwae_first = rdRun(swwae_first,I);

    figure;
    clf
    subplot(1,3,1);
    imshow( I );
    title( 'input' );
    subplot(1,3,2);
    imshow( R_sae_first );
    title( sprintf( '%s-based\nSAE-first\n(5th layer)', base_networks{b}) );
    subplot(1,3,3);
    imshow( R_swwae_first );
    title( sprintf( '%s-based\nSAE-first\n(5th layer)', base_networks{b}) );

end

% cls + reconstruction
%swwae_all   = rdNet('vggnet/cls/SWWAE-all/layer5');
%sae_all     = rdNet('vggnet/cls/SAE-all/layer5');

