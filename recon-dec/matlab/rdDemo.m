function rdDemo

% add neccessary paths
RDMAT_ROOT = fileparts( mfilename('fullpath') );
addpath(RDMAT_ROOT);
rdInit;

% reconstruction
swwae_first = rdNet('vggnet/recon/SWWAE-first/layer5');
sae_first   = rdNet('vggnet/recon/SAE-first/layer5');

% cls + reconstruction
swwae_all   = rdNet('vggnet/cls/SWWAE-all/layer5');
sae_all     = rdNet('vggnet/cls/SAE-all/layer5');

