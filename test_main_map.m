%   'test_main.m' is the main evaluation script for testing vairous color
%   transfer approximation methods. The error between the original color
%   transfer output and an approximated one is measured in PSNR an SSIM.
%   Also, we show a one-way anova post hoc test to analyse the similarity
%   between different results.

%   Copyright 2018 Han Gong <gong@fedoraproject.org>, University of East
%   Anglia.

%   References:
%   Gong, H., Finlayson, G.D., Fisher, R.B. and Fang, F., 2017. 3D color
%   homography model for photo-realistic color transfer re-coding. The
%   Visual Computer, pp.1-11.

% configuration
ref = './cc/xrite_cc_ref_swatch.PNG';
source = './cc/DJI_CC_01.CCchip.means.PNG';
original = './cc/DJI_CC_01';

% define colour enhancement methods
ap.Name = {'3D_H'}; % name of the methods.
Nap = numel(ap.Name); % number of approximation methods

% discover all images
ref_img     = im2double(imread(ref)); % ref swatch
source_img  = im2double(imread(source)); % original image swatch
original_img_path = sprintf('%s.JPG',original)
original_img = im2double(imread(original_img_path));

for i_ap = 1:Nap
    ap_h = str2func(['cf_',ap.Name{i_ap}]); % function handle
    [img,model] = ap_h(source_img,ref_img);
    f_ap = sprintf('%s.CC_%s.PNG',original,ap.Name{i_ap});
    imgr = reshape(original_img,[],3);
    % re-apply to a higher res image
    sz = size(original_img);
    cc = model.H*[imgr';ones(1,size(imgr,1))];
    cc = bsxfun(@rdivide,cc(1:3,:),cc(4,:));
    cc = min(max(cc,0),1);
    n = size(cc,2);
    meancc = mean(cc,1)'; % transformed brightness
    meanf = model.pp(1+floor(meancc*999));
    meanf = max(meanf,0);
    nd = meanf./meancc; % convert brightness change to shading
    nd(meancc<1/255) = 1; % discard dark pixel shadings
    D = sparse(1:n,1:n,nd(:),n,n);
    ei = reshape(cc',sz);
    ImD = full(reshape(diag(D),sz(1:2)));
    %if opt.use_denoise % denoise the shading field
    %    grey = rgb2gray(oi);
    %    ImD = bFilter(ImD,grey,0,1,12);
    %end
    ei = min(max(ei.*ImD,0),1);
    imwrite(ei, f_ap);
end