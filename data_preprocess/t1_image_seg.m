if ~license('test', 'Distrib_Computing_Toolbox')
    error('Parallel Computing Toolbox is not available.');
end

% add SPM path
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% t1 path
main_dir = " ";

% get subfolders
sub_folders = dir(main_dir);
sub_folders = sub_folders([sub_folders.isdir]);
sub_folders = sub_folders(~ismember({sub_folders.name}, {'.', '..'}));

% initalize
poolobj = gcp('nocreate'); 
if isempty(poolobj)
    poolobj = parpool;
end

% batch
batch_size = 7;

num_batches = ceil(length(sub_folders) / batch_size);

for batch_idx = 1:num_batches

    start_idx = (batch_idx - 1) * batch_size + 1;
    end_idx = min(batch_idx * batch_size, length(sub_folders));
    

    indices = start_idx:end_idx;

    results = cell(1, numel(indices));

    parfor sub_idx = indices
        try
           
            sub_folder_path = fullfile(main_dir, sub_folders(sub_idx).name);
            disp(sub_folder_path)
            
            result = processSubFolder(sub_folder_path);
            
            results{sub_idx} = result;
            
            disp(['Processing sub_folder_path: ', sub_folder_path]);
            
        catch ME
            warning('wrong: %s', sub_folders(sub_idx).name, ME.message);
        end
    end
end

function result = processSubFolder(sub_folder_path)

    sub_folder_path = char(sub_folder_path);
    % check the end of  'Crop_1.nii' 
    nii_files = dir(fullfile(sub_folder_path, '*Crop_1.nii'));

    if isempty(nii_files)
        warning('未找到文件：%s', fullfile(sub_folder_path, '*Crop_1.nii'));
    end

    % get  t1_path
    t1_image = char(fullfile(sub_folder_path, nii_files(1).name));
    % t1_image = '  ';
    t1_image_path = spm_file(t1_image, 'path');

    clear matlabbatch;
    %  (Segmentation)
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {t1_image};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,1'};
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,2'};
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,3'};
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,4'};
    matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,5'};
    matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {'E:\Tools\DPABI\spm12\tpm\TPM.nii,6'};
    matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 1];

    spm_jobman('run', matlabbatch);
    clear matlabbatch;

    % ------------------------

    file_list = dir(t1_image_path);
    pattern = '^c1.*\.nii$';
    grey_matter_files = {};
    for i = 1:length(file_list)
        if ~file_list(i).isdir && ~isempty(regexp(file_list(i).name, pattern, 'once'))
            grey_matter_files{end+1} = fullfile(t1_image_path, file_list(i).name);
        end
    end
    if isempty(grey_matter_files)
        error('Grey matter file not found: c1*.nii');
    else
        grey_matter = grey_matter_files{1};
    end

    pattern = '^c2.*\.nii$';
    white_matter_files = {};
    for i = 1:length(file_list)
        if ~file_list(i).isdir && ~isempty(regexp(file_list(i).name, pattern, 'once'))
            white_matter_files{end+1} = fullfile(t1_image_path, file_list(i).name);
        end
    end
    if isempty(white_matter_files)
        error('White matter file not found: c2*.nii');
    else
        white_matter = white_matter_files{1};
    end

    pattern = '^c3.*\.nii$';
    csf_files = {};
    for i = 1:length(file_list)
        if ~file_list(i).isdir && ~isempty(regexp(file_list(i).name, pattern, 'once'))
            csf_files{end+1} = fullfile(t1_image_path, file_list(i).name);
        end
    end
    if isempty(csf_files)
        error('csf file not found: c3*.nii');
    else
        csf = csf_files{1};
    end

    matlabbatch{1}.spm.tools.dartel.warp.images = {
        {grey_matter}
        {white_matter}
    };
    matlabbatch{1}.spm.tools.dartel.warp.settings.rform = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).rparam = [4 2 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(1).slam = 16;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).rparam = [2 1 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).K = 0;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(2).slam = 8;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).rparam = [1 0.5 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).K = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(3).slam = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).rparam = [0.5 0.25 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).K = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(4).slam = 2;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).rparam = [0.25 0.125 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).K = 4;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(5).slam = 1;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).its = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).rparam = [0.25 0.125 1e-06];
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).K = 6;
    matlabbatch{1}.spm.tools.dartel.warp.settings.param(6).slam = 0.5;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.lmreg = 0.01;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.cyc = 3;
    matlabbatch{1}.spm.tools.dartel.warp.settings.optim.its = 3;


    spm_jobman('run', matlabbatch);
    clear matlabbatch;


    matlabbatch{1}.spm.tools.dartel.mni_norm.template = {fullfile(spm_file(t1_image,'path'), 'Template_6.nii')};


    flow_field_dir = spm_file(t1_image, 'path');
    files = dir(flow_field_dir);
    pattern = '^u_c1.*\.nii$';
    flow_field_files = {};
    for i = 1:length(files)
        if ~files(i).isdir && ~isempty(regexp(files(i).name, pattern, 'once'))
            flow_field_files{end+1} = fullfile(flow_field_dir, files(i).name);
        end
    end
    if isempty(flow_field_files)
         error('Flow field file not found: u_c1*.nii');
    else
        flow_field = flow_field_files{1};
    end
    % flow_field = fullfile(spm_file(t1_image, 'path'), 'u_rc1sub_0001_MPRAGE_SENSE_20110602075636_301_Crop_1_Template.nii');

    matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = {flow_field};
    matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images = {
        {t1_image}
        {grey_matter}
        {white_matter}
        {csf}
        };
    matlabbatch{1}.spm.tools.dartel.mni_norm.vox = [1 1 1];
    matlabbatch{1}.spm.tools.dartel.mni_norm.bb = [-78 -112 -70
                                                   78 76 85];
    matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = 0;
    % matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [8 8 8];   % 平滑
    matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm = [0 0 0];

    % noramilization
    spm_jobman('run', matlabbatch);
    clear matlabbatch;

    disp(['DARTEL到MNI标准化完成: ', sub_folder_path]);
    
    w_file_list = dir(t1_image_path);
    w_pattern = '^wc1.*\.nii$';
    w_grey_matter_files = {};
    for i = 1:length(w_file_list)
        if ~w_file_list(i).isdir && ~isempty(regexp(w_file_list(i).name, w_pattern, 'once'))
            w_grey_matter_files{end+1} = fullfile(t1_image_path, w_file_list(i).name);
        end
    end
    if isempty(w_grey_matter_files)
        error('Grey matter file not found: wc1*.nii');
    else
        wc1_path = w_grey_matter_files{1};
    end
    % 使用正则表达式匹配wc2*.nii文件
    w_pattern = '^wc2.*\.nii$';
    w_white_matter_files = {};
    for i = 1:length(w_file_list)
        if ~w_file_list(i).isdir && ~isempty(regexp(w_file_list(i).name, w_pattern, 'once'))
            w_white_matter_files{end+1} = fullfile(t1_image_path, w_file_list(i).name);
        end
    end
    if isempty(w_white_matter_files)
        error('White matter file not found: wc2*.nii');
    else
        wc2_path = w_white_matter_files{1};
    end
    % 使用正则表达式匹配wc3*.nii文件
    w_pattern = '^wc3.*\.nii$';
    w_csf_files = {};
    for i = 1:length(w_file_list)
        if ~w_file_list(i).isdir && ~isempty(regexp(w_file_list(i).name, w_pattern, 'once'))
            w_csf_files{end+1} = fullfile(t1_image_path, w_file_list(i).name);
        end
    end
    if isempty(w_csf_files)
        error('csf file not found: wc3*.nii');
    else
        wc3_path = w_csf_files{1};
    end
    % 使用正则表达式匹配wsub*.nii文件
    w_pattern = '^wsub.*\.nii$';
    wsub_files = {};
    for i = 1:length(w_file_list)
        if ~w_file_list(i).isdir && ~isempty(regexp(w_file_list(i).name, w_pattern, 'once'))
            wsub_files{end+1} = fullfile(t1_image_path, w_file_list(i).name);
        end
    end
    if isempty(wsub_files)
        error('csf file not found: wc3*.nii');
    else
        wsub_path = wsub_files{1};
    end

    % load aal.nii
    aal_vol = spm_vol(aal_atlas_path);
    aal_data = spm_read_vols(aal_vol);
    wc1_vol = spm_vol(wc1_path);
    wc1_data = spm_read_vols(wc1_vol);

    wc2_vol = spm_vol(wc2_path);
    wc2_data = spm_read_vols(wc2_vol);

    wc3_vol = spm_vol(wc3_path);
    wc3_data = spm_read_vols(wc3_vol);

    wsub_vol = spm_vol(wsub_path);
    wsub_data = spm_read_vols(wsub_vol);
    aal_labels = unique(aal_data(:));
    aal_labels(aal_labels == 0) = []; 
    num_rois = length(aal_labels);
    features_gm = zeros(num_rois, 1);
    features_wm = zeros(num_rois, 1);
    features_csf = zeros(num_rois, 1);
    features_sub = zeros(num_rois, 1);
    for i = 1:num_rois
        roi_label = aal_labels(i);
        roi_mask = (aal_data == roi_label);
        gm_volume = sum(wc1_data(roi_mask));
        features_gm(i) = gm_volume;
        wm_volume = sum(wc2_data(roi_mask));
        features_wm(i) = wm_volume;
        csf_volume = sum(wc3_data(roi_mask));
        features_csf(i) = csf_volume;
        sub_volume = sum(wsub_data(roi_mask));
        features_sub(i) = sub_volume;
    end

    disp('Gray Matter Volume Features:');
    disp(features_gm);

    disp('White Matter Volume Features:');
    disp(features_wm);

    disp('CSF Volume Features:');
    disp(features_csf);

    disp('SUB Volume Features:');
    disp(features_sub);

    save(fullfile(t1_image_path, 'features_gm.mat'), 'features_gm');
    save(fullfile(t1_image_path, 'features_wm.mat'), 'features_wm');
    save(fullfile(t1_image_path, 'features_csf.mat'), 'features_csf');
    save(fullfile(t1_image_path, 'features_sub.mat'), 'features_sub');
    
    result = sub_folder_path;
end