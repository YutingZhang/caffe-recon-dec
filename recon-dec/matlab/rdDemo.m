function rdDemo

RDMAT_ROOT = fileparts( mfilename('fullpath') );
addpath(RDMAT_ROOT);

rdInit;

swwae_first = rdNet('vggnet/recon/SWWAE-first/layer5');
sae_first   = rdNet('vggnet/recon/SAE-first/layer5');
swwae_all   = rdNet('vggnet/cls/SWWAE-all/layer5');
sae_all     = rdNet('vggnet/cls/SAE-all/layer5');

