function data=dynamic_lungwater_pipeline(data)

pathname = 'C:\Users\User\Documents\folder\'; %paste your local path of the pyhton lung segmentation network here
tf=1; %timeframe

disp('Lung segmentation');
data = lung_segmentation(data,tf,pathname);

disp('Body mask');
data = body_roi(data,tf);

disp('Coil shading correction');
data = coil_shading_correction(data,tf, data.image_moco_uncorr); %shading map calculated in first timeframe only

disp('LWD calculation');
data = calc_lwd(data);

disp('Post-processing finished.');
end



function data = lung_segmentation(data,tf, pathname)


script=[pathname, 'compute_lung_seg2022.py'];
input=[pathname, 'image_ML.mat'];
output=[pathname, 'mask_ML.mat'];
command = sprintf('python3 %s %s %s', script,input,output);


resolution=data.resolution;

im=squeeze(data.image_moco(:,:,:,tf));
im=imresize(im, round(resolution(1)/1.5), 'bicubic'); %upsample to ~1.5mm resolution
save(input, 'im');

system(command); %run network
load(output);

mask=imresize(mask, 1/round(resolution(1)/1.5), 'bicubic'); %downsample
mask(mask>=0.3)=1;
mask(mask<0.3)=0;

data.lungMask(:,:,:,1:size(data.image_moco,4))=repmat(mask, [1,1, 1, size(data.image_moco,4)]);
data.lungVolume=sum(mask(:))*data.resolution(1)*data.resolution(2)*data.resolution(3)/1000; %in ml

end


function data = body_roi(data,tf)
data.bodyMask=[];
mask=squeeze(data.lungMask(:,:,:,tf));

se = strel('sphere',4);
void(:,:,:,1) = imdilate(mask, se);

se = strel('sphere',8);
body(:,:,:,1) = imdilate(mask, se);

body=body-void;

noBodySlices=1:size(mask,3);
noBodySlices(round(size(mask,3)/2)-2:round(size(mask,3)/2)+2)=[]; %five central slices
body(:,:,noBodySlices,1)=0*body(:,:,noBodySlices);

data.bodyMask(:,:,:,1:size(data.image_moco,4))=repmat(single(body), [1,1, 1, size(data.image_moco,4)]);
end

function data = coil_shading_correction(data,tf, image)


shading_map=[];


im_in=squeeze(image(:,:,:,tf));

bodymask=zeros(size(im_in));
bodymask(im_in>=mean(im_in(:)))=1;

bodymask(squeeze(data.lungMask(:,:,:,tf))==1)=0; %exclude lungs

im = (im_in-min(im_in(:)))./max((im_in(:)-min(im_in(:)))); %normalize for 0-1
im(isnan(im))=0; im(isinf(im))=0;

if tf==1
    midSlice=round(size(im,3)/2);
    lambda= findLambdaSmoothing(im(:,:,midSlice));
    data.lambda=lambda;
end

for loop=1:size(image,3) %slice-by-slice Tikhonov regularization
    shading_map(:,:,loop,tf) = tikReg2D(im(:,:,loop).*bodymask(:,:,loop),lambda);
end


data.shading_map(:,:,:,1:size(image,4))=repmat(shading_map, [1,1, 1, size(image,4)]);

%normalize image
data.image_norm = data.image_moco./data.shading_map; %apply shading bask to jacobian corrected image
data.image_norm(isinf(data.image_norm))=0;
data.image_norm(isnan(data.image_norm))=0;




end


function lambda = findLambdaSmoothing(image, lambda_range)


if nargin < 2
    lambda_range=linspace(0.5,2000); %lambda search range
else
    lambda_range=linspace(lambda_range(1),lambda_range(2)); %lambda search range
end
image=double(image);

ind = image>0; %values less than zero will be excluded from the fitting process
b = image(image>0);


for i = 1:length(lambda_range)
    %Fitting the surface with a specific smoothing parameter
    [x,A,T] = tikReg2D(image,lambda_range(i));
    
    %calculating the errors with that parameter
    res_norm(i) = norm(A*x(:)-b,'fro');
    solution_norm(i) = norm(T*x(:),'fro');
end

res_norm_log = log(res_norm);
solution_norm_log = log(solution_norm);

x_grid = 0.5:0.25:50000;
%interpolate norms
res_norm_log= spline(lambda_range,res_norm_log,x_grid);
solution_norm_log = spline(lambda_range,solution_norm_log,x_grid);

%calculating maximum curvature, derivatives
xL1 = gradient(res_norm_log);
yL1 = gradient(solution_norm_log);

xL2 = del2(res_norm_log);
yL2 = del2(solution_norm_log);

k = (xL2.*yL1-xL1.*yL2)./(xL1.^2+yL1.^2).^1.5; %curvature equations
[~,ind] = min(k);
lambda = x_grid(ind); %optimized lambda at max curvature

end

function [X,A,T] = tikReg2D(image,lambda)
% tikReg2D() generates a surface to fit to the data in "slice". Based on
% John D'Errico's Gridfit. Zeros are ignored from the fit, tikhonov
% regularization allows for fitting a surface over data with large holes or
% missing data.
%
% slice == data to fit a surface over (will fit to non-zero data in the
% array), 2D matrix
% lambda == smoothing paramter, the higher the value, the smoother the fit
%
%Tikhonov regularizer
%min(Ax-b+(lambda)*Tx)

image=double(image);
[ny, nx, nz] = size(image);

b = image(image(:)>0); %rhs data, assuming 0 values are to be excluded from the fit
bind = find(image(:)); %rhs location in full grid

nb = length(b);
ngrid = length(image(:));

%Holds the information for the location of each b value in the full grid
%(bInd) while having a row corresponding to each b value.
A = sparse((1:nb)',bind, ones(nb,1),nb,ngrid);


%difference approximation in y
[i,j] = meshgrid(1:nx,2:(ny-1));
ind = j(:) + ny*(i(:)-1);
len = length(ind);

T2 = sparse(repmat(ind,1,3), [ind-1,ind,ind+1], [-1*ones(len,1),2*ones(len,1),-1*ones(len,1)], ngrid,ngrid);


%difference approximation in x
[i,j] = meshgrid(2:(nx-1),1:ny);
ind = j(:) + ny*(i(:)-1);
len = length(ind);

T1 = sparse(repmat(ind,1,3), [ind-ny,ind,ind+ny], [-1*ones(len,1),2*ones(len,1),-1*ones(len,1)], ngrid,ngrid);

%Combining regularization (tikhonov) matrices
T = [T1;T2];

%appending zeros to the rhs
b = [b;zeros(size(T,1),1)];

%solving the minimization problem (tikhonov regularization solution)
AT = [A;lambda*T];
X = reshape((AT'*AT)\(AT'*b),ny,nx);
end


function data = calc_lwd(data)

for tf= 1:size(data.image_moco,4)
    im=squeeze(data.image_norm(:,:,:,tf));
    LWD_map = 70*im./median(im(squeeze(data.bodyMask(:,:,:,tf))==1)); %assuming hepatic water density is 70%
    data.LWD(tf) = mean(LWD_map(squeeze(data.lungMask(:,:,:,tf))==1));
    data.LWD_map(:,:,:,tf)=LWD_map;
end
data.DeltaLWD = 100*(data.LWD-data.LWD(1))/data.LWD(1); %percentual change in LWD
end




