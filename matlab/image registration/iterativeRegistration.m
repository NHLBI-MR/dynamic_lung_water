function [output] = iterativeRegistration(images_binned, ref)

if nargin <2
    ref = 2; %stable end-expiratory phase
end

s=size(images_binned{1}{1}{1});
numbins=s(end);
tfs=size(images_binned{1},2);
s(end)=numbins*tfs;
images = zeros(s);
counter=1;

%rotate images so they are not upside down
for ii=1:tfs
    ims = rot90(abs(images_binned{1}{ii}{1}),2);
    for jj=1:numbins
        images(:,:,:,counter)=ims(:,:,:,jj);
        counter=counter+1;
    end
end

%image registration
output = register_images_RTT(images, ref);

%jacobian correction
images_jac = output.images.*output.jacobian; 


%average over bins
counter=1;
for ii=1:numbins:numbins*tfs
    images_moco(:,:,:,counter)=mean(images_jac(:,:,:,ii:ii+numbins-1), 4); %average all bins into one timeframe
    images_moco_smooth(:,:,:,counter)=mean(images_jac_smooth(:,:,:,ii:ii+numbins-1), 4); %average all bins into one timeframe
    images_uncorr(:,:,:,counter) = mean(output.images(:,:,:,ii:ii+numbins-1), 4);
    counter=counter+1;
end

output.images_resp_res=images;
output.images_moco=images_moco;
output.images_moco_smooth=images_moco_smooth;
output.jacobian_smooth=jac;
output.images_uncorr=images_uncorr;

end
