function data=ute_slwindow_moco_recon(filename, numbins, durationImage, slWindowIncr)


if nargin<2
    numbins=8; %number of bins to sort respiratory position into
    durationImage=1.5; %amount of sampled data per timeframe, in minutes
    slWindowIncr=1/3; %sliding window temporal increment, in minutes
end

%%


disp('Reading Data')
[kspace_full,traj_full,header] = data_read_spiral_data_stream_h5(filename,[],0);

kspace = kspace_full;
traj = traj_full;
kspace.kspace = reshape (kspace_full.kspace,[],header.channels(1));
traj.corrected = reshape (traj_full.corrected(1:size(kspace.dc_sig,1),:,:),[],3);

% Find combination weights for concomitant field corrections using MFI
disp('Estimatring Concomitant Fields')
[C,demodulation_freqs,scaled_time,frq_plot] = concomitant_field_correction(header,traj.grad_uncorrected);

%Estimate rotation angle for each interleave
allangles=rad2deg(angle(traj.grad_uncorrected(25,:,1)+1j*traj.grad_uncorrected(25,:,2)));

%% gating

disp('Estimating Respiratory Signal')

user_opts.numBins = numbins;
user_opts.evenBins = true;
user_opts.stableBinning = false;

nav_gating = selfGating_lit(kspace.nav_data,header,allangles,user_opts);
plot(nav_gating.timestamp,nav_gating.gating_signal);

totalTime=(size(kspace.kspace,1)+size(kspace.nav_data,2))*(header.sequenceParameters.TR * 1e-3)/60; %in minutes

%define temporal intervals for bins
if totalTime>=durationImage
    nav_image_size = floor(size(nav_gating.gating_signal,2)/(totalTime/durationImage));
    nav_step_size= floor(size(nav_gating.gating_signal,2)/(totalTime/slWindowIncr));
    
    maxInt=nav_image_size:nav_step_size:(totalTime/durationImage)*nav_image_size;
    minInt=1:nav_step_size:((totalTime/durationImage)*nav_image_size);
    minInt=minInt(1:length(maxInt));
    
    nbr_of_timeframes=length(maxInt);
else
    minInt=1;
    maxInt=floor(size(nav_gating.gating_signal,2));
    nbr_of_timeframes=1;
end

clear images_binned averaged_image

for ii = 1:nbr_of_timeframes %loop over all timeframes to reconstruct
    clear averagedOut;
    
    %extract and store data for timeframe to reconstruct
    idx=minInt(ii):maxInt(ii);
    selectedSig = nav_gating.gating_signal(idx);
    ds=nav_gating.timestamp(idx);
    
    output = binning(selectedSig,user_opts.numBins,user_opts.evenBins);
    
    for bins=1:numbins
        struct.accepted_times{ii}{bins}= ds(find(selectedSig<output(2,bins) & selectedSig>output(1,bins)));
        idx_to_send_nav_cell{ii}{bins} = get_idx_to_send(header.dataacq_time_stamp,struct.accepted_times{ii}{bins},nav_gating.sampling_time);
    end
    
    idx_to_send = cell2mat(idx_to_send_nav_cell{ii}');
    averagedOut.kspace  = kspace.kspace(idx_to_send,:);
    averagedOut.traj    = traj.corrected(idx_to_send,:);
    averagedOut.sct    = scaled_time(:,idx_to_send);
    averagedOut.shots_per_time = cellfun(@length,idx_to_send_nav_cell{ii});
    
    disp('Reconstructing Binned Data Sense')
    user_opts.csm                   = [];
    user_opts.header                = header;
    user_opts.mode                  = 0 ;
    user_opts.filterType            = '';
    user_opts.doSense               = true;
    user_opts.cweights              = C;
    user_opts.demodulation_frequency= demodulation_freqs;
    user_opts.doTV                  = false;
    user_opts.lambda_sp             = 0.1; %spatial constraint
    user_opts.lambda_ti             = 0.1; %temporal contrtraint
    user_opts.iterations            = 10; %number of iterations for reconstruction
    user_opts.shots_per_time        =averagedOut.shots_per_time; %amount of data in each bin
    user_opts.calcCSM               = true;
    user_opts.runlocal = 0;
    
    if(iscell(averagedOut.sct))
        user_opts.scaled_time           = reshape(cell2mat(averagedOut.sct),size(averagedOut.sct{1},1),[]);
    else
        user_opts.scaled_time           = averagedOut.sct;
    end
    user_opts.demodulation_frequency= demodulation_freqs(1);
    user_opts.cweights              = ones(size(C,1),size(C,2),size(C,3),1);
    user_opts.reconType             = 3; %spatial and temporal total variation 4D without concominant field correction
    
    [images_binned{1}{ii},~] = data_reconstruct_gt(averagedOut.kspace,averagedOut.traj,user_opts); %run recon
end


disp('Image Registration')
[output] = iterativeRegistration(images_binned);

clear data
data.image_moco=output.images_moco; %jacobian corrected images for quantification
data.image_moco_uncorr=output.images_uncorr; %no jacobian correction

data.numbins=numbins;
data.nbr_of_timeframes=nbr_of_timeframes;
data.durationImage=durationImage;
data.slWindowIncr=slWindowIncr;
data.totalTime=totalTime;
data.time = linspace(0,slWindowIncr*(nbr_of_timeframes-1),nbr_of_timeframes);
data.resolution =    round([user_opts.header.encoding.encodedSpace.fieldOfView_mm.x user_opts.header.encoding.encodedSpace.fieldOfView_mm.y user_opts.header.encoding.encodedSpace.fieldOfView_mm.z]./[user_opts.header.encoding.encodedSpace.matrixSize.x user_opts.header.encoding.encodedSpace.matrixSize.y user_opts.header.encoding.encodedSpace.matrixSize.z],1);

%post-processing
data=dynamic_lungwater_pipeline(data);

disp('Saving mat-file')
save([filename(1:end-3), '_',num2str(numbins), '_bins_', num2str(durationImage), '_incr_',num2str(slWindowIncr), 'min_MOCO_RECON.mat'],'-v7.3','data');

end

