function [reconImg, predLabel, inputImg] = rdRun( netx, inputImg )

im_data = inputImg;
if size(im_data, 3) == 3
    im_data = im_data(:, :, [3, 2, 1], : );
end
% flip width and height to make width the fastest dimension
im_data = permute(im_data, [2, 1, 3, 4]);
% convert from uint8 to single
im_data = single(im_data);

im_data = imresize( im_data, netx.geo_shape );
im_data = bsxfun( @minus, im_data, netx.mean );

netx.net.blobs(netx.im_input).set_data(im_data);
netx.net.forward_prefilled();

if isempty(netx.im_output)
    reconImg = [];
else
    reconImg = netx.net.blobs(netx.im_output).get_data();
    reconImg = bsxfun( @plus, reconImg, netx.mean/255 );
    reconImg = permute(reconImg, [2, 1, 3, 4]);
    reconImg = reconImg(:, :, [3, 2, 1], : );
end

if isempty(netx.label_output)
    predLabel = [];
else
    predScores = netx.net.blobs(netx.label_output).get_data();
    [~,predLabel] = max(predScores);
end

inputImg = im_data./255;
inputImg = bsxfun( @plus, inputImg, netx.mean/255 );
inputImg = permute(inputImg, [2, 1, 3, 4]);
inputImg = inputImg(:, :, [3, 2, 1], : );
