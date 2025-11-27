if ~license('test', 'Distrib_Computing_Toolbox')
    error('Parallel Computing Toolbox is not available.');
end

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% root path
main_dir = '  ';

% AAL path
aal_atlas_path = 'E:\Tools\aal_mdd.nii';

sub_folders = dir(main_dir);
sub_folders = sub_folders([sub_folders.isdir]);
sub_folders = sub_folders(~ismember({sub_folders.name}, {'.', '..'}));

poolobj = gcp('nocreate');
if isempty(poolobj)
    poolobj = parpool;
end

batch_size = 12;
num_batches = ceil(length(sub_folders) / batch_size);

for batch_idx = 1:num_batches
    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx   = min(batch_idx * batch_size, length(sub_folders));
    indices   = start_idx:end_idx;
    
    parfor k = indices
        try
            sub_path = fullfile(main_dir, sub_folders(k).name);
            disp(['Processing AAL features: ', sub_path]);
            extractAALfeatures(sub_path, aal_atlas_path);
            disp(['Done: ', sub_folders(k).name]);
        catch ME
            warning('Error in %s: %s', sub_folders(k).name, ME.message);
        end
    end
end

function extractAALfeatures(sub_folder_path, aal_atlas_path)
    
    wc1 = dir(fullfile(sub_folder_path, 'wc1*.nii'));
     %wc2 = dir(fullfile(sub_folder_path, 'wc2*.nii'));
     %wc3 = dir(fullfile(sub_folder_path, 'wc3*.nii'));
    %if isempty(wc1) || isempty(wc2) || isempty(wc3)
    if isempty(wc1)
        error('缺少 wc1/wc2/wc3/wsub 文件');
    end
    wc1_path = fullfile(sub_folder_path, wc1(1).name);
     %wc2_path = fullfile(sub_folder_path, wc2(1).name);
     %wc3_path = fullfile(sub_folder_path, wc3(1).name);

    aal_vol  = spm_vol(aal_atlas_path);
    aal_data = spm_read_vols(aal_vol);

    gm_vol  = spm_vol(wc1_path); gm_data  = spm_read_vols(gm_vol);
     %wm_vol  = spm_vol(wc2_path); wm_data  = spm_read_vols(wm_vol);
     %csf_vol = spm_vol(wc3_path); csf_data = spm_read_vols(csf_vol);

    labels = unique(aal_data(:));
    labels(labels == 0) = [];

    nROI = numel(labels);
    feats_gm  = zeros(nROI,1);
     %feats_wm  = zeros(nROI,1);
     %feats_csf = zeros(nROI,1);

    for i = 1:nROI
        roi = (aal_data == labels(i));
        feats_gm(i)  = sum(gm_data(roi));
         %feats_wm(i)  = sum(wm_data(roi));
         %feats_csf(i) = sum(csf_data(roi));
    end

    save(fullfile(sub_folder_path, 'features_gm.mat' ), 'feats_gm');
     %save(fullfile(sub_folder_path, 'features_wm.mat' ), 'feats_wm');
     %save(fullfile(sub_folder_path, 'features_csf.mat'), 'feats_csf');
end
