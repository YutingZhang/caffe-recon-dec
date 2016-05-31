function rdInit

RDMAT_ROOT = fileparts( mfilename('fullpath') );
CAFFE_ROOT = fileparts( fileparts( RDMAT_ROOT ) );

addpath(RDMAT_ROOT);
addpath(genpath(fullfile(CAFFE_ROOT,'matlab')));

