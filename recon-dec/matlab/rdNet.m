function netx = rdNet( model_name )

RD_ROOT = fileparts(fileparts( mfilename('fullpath') ));

% locate model
PROTOTXT_PATH = fullfile( RD_ROOT, [model_name '_deploy.prototxt'] );
WEIGHT_PATH   = fullfile( RD_ROOT, [model_name '.caffemodel'] );
assert( exist(PROTOTXT_PATH,'file')~=0, 'No such model' );

% fetch network from the web weights if necessary
if ~exist( WEIGHT_PATH, 'file' )
    fprintf('Fetch model: %s\n', model_name);
    [~,~] = system( sprintf('%s/fetch_model.sh "%s"', RD_ROOT, model_name) );
end

% init network
netx = [];
netx.net = caffe.Net(PROTOTXT_PATH, WEIGHT_PATH, 'test' );
netx.on_cleanup = onCleanup( @() netx.net.reset() );

% set up mean if necessary

first_token_idx = find(model_name=='/',1);
base_network = model_name(1:first_token_idx);

switch base_network
    case 'vggnet'
        netx.mean = reshape( single([103.939 116.779 123.68]), [1 1 3] );
    case 'alexnet'
        netx.mean = load(fullfile(RD_ROOT,'../matlab/+caffe/imagenet/ilsvrc_2012_mean.mat'));
        netx.mean = netx.mean.mean_data;
    otherwise
end
data_shape=netx.net.blobs('data').shape();
mean_shape=size(netx.mean);
mean_shape=[mean_shape,ones(1,length(data_shape)-length(mean_shape))];
rep_factor = (data_shape./mean_shape);
assert( round(rep_factor)==rep_factor, 'mean does not match with data' );
netx.mean = repmat( netx.mean, rep_factor );

input_blob_names = netx.net.inputs;
if ismember( 'mean', input_blob_names );
    netx.net.blobs('mean').set_data( netx.mean );
end

netx.im_input     = 'data';
netx.im_output    = 'dec:data';
netx.label_output = 'fc8';
netx.geo_shape    = data_shape(1:2);

blob_names = netx.net.blob_names;

if ~ismember(netx.im_output,blob_names)
    netx.im_output = [];
end

if ~ismember(netx.label_output,blob_names)
    netx.label_output = [];
end

