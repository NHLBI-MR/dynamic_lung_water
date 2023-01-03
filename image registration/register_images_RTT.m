function [output] = register_images_RTT(images,referenceNumber,alpha)

% images - [X Y Z t]
% referenceNumber - index of the reference image

if(nargin<2)
    referenceNumber=2;
end
if(nargin<3)
    alpha = 9; % (use 4-32 lowest value that gives the best value)
    alpha = 32; % testing for HV25
end

images = double(images);

minVal = min(images(:));
images = images - minVal;
maxVal = max(images(:));
images = images ./ maxVal;
images = images * 2048;
deformed_image = zeros(size(images));
def = zeros(size(images,1),size(images,2),size(images,3),3, size(images,4));


tic


% Define registration parameters
ref = referenceNumber;
tic;


sizeMat =[size(images,1), size(images,2), size(images,3)];

for ii=1:size(images,4) 
    ii
    clear RTTrackerWrapper;
    Iref = images(:,:,:,ref);
    
    RTTrackerWrapper(sizeMat(1),sizeMat(2),sizeMat(3), ...
        2, ...
        0, ...
        0, ...
        alpha); %Smoothing parameter
    
    RTTrackerWrapper(abs(Iref), abs(images(:,:,:,ii)));
    
    [deformed_image(:,:,:,ii)] = RTTrackerWrapper(images(:,:,:,ii));
    def(:,:,:,:,ii) = RTTrackerWrapper();
    jacob(:,:,:,ii) = estimateJacobian(def(:,:,:,:,ii));
end

deformed_image = deformed_image / 2048;
deformed_image = deformed_image * maxVal;
deformed_image = deformed_image + minVal;

output.images = single(deformed_image);
output.deformation = single(def);
output.jacobian = single(jacob);

end

function jacobian = estimateJacobian(deformation)
% Estimate jacobian determinant
% Author: Ahsan Javed (11/22/2021)

%Calculate Gradients X,Y,Z
[gradxVx,gradyVx,gradzVx] = gradient(deformation(:,:,:,2),1,1,1);
[gradxVy,gradyVy,gradzVy] = gradient(deformation(:,:,:,1),1,1,1);
[gradxVz,gradyVz,gradzVz] = gradient(deformation(:,:,:,3),1,1,1);



gradV = zeros(3,3,size(deformation,1),size(deformation,2),size(deformation,3));

for i=1:size(deformation,1)
    for j=1:size(deformation,2)
        for k=1:size(deformation,3)
            gradV(1,1,i,j,k) = gradxVx(i,j,k);
            gradV(2,1,i,j,k) = gradxVy(i,j,k);
            gradV(3,1,i,j,k) = gradxVz(i,j,k);
            
            gradV(1,2,i,j,k) = gradyVx(i,j,k);
            gradV(2,2,i,j,k) = gradyVy(i,j,k);
            gradV(3,2,i,j,k) = gradyVz(i,j,k);
            
            gradV(1,3,i,j,k) = gradzVx(i,j,k);
            gradV(2,3,i,j,k) = gradzVy(i,j,k);
            gradV(3,3,i,j,k) = gradzVz(i,j,k);
        end
    end
end

% Estimating jacobian determinant
temp = reshape(gradV,3,3,[]);
jacobian = zeros(size(temp,3),1);
for ii=1:size(temp,3)
    jacobian(ii) = det(eye(3)+temp(:,:,ii));
end

jacobian = reshape(jacobian,size(deformation,1),size(deformation,2),size(deformation,3));
end