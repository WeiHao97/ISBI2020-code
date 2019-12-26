clear all
addpath('./altmany-export_fig-76bd7fa/');

% This is a mapping from the IIT atlas indices to TADPOLE indices.
% Probably not necessary for others' code.
load iit_to_tadpole_idx.mat

% N x T matrix: N-dimensional vector of T time points.
av45_dx_gen = rand(82, 3);
%% Interpolating between the time points for smooth images.
T = 3;
num_points = 11;
av45_interp = av45_dx_gen(:, 1);
T_interp = 1;
for t=1:T-1
    av45_T1 = av45_dx_gen(:, t);
    av45_T2 = av45_dx_gen(:, t+1);
    av45_diff = av45_T2 - av45_T1;
    
    tt = linspace(t, t+1, num_points);
    scale = 1 / (num_points - 1);
    
    av45_interp_delta = scale * av45_diff;
    
    for j=1:num_points-1
        av45_interp = [av45_interp, av45_interp(:,end) + av45_interp_delta];
        T_interp = [T_interp, T_interp(end) + scale];
    end
end


%% This loop produces top and front views. The side view is shown below
% this block which is nearly identical.
close all;
% Desikan atlas 3D surface.
load glass_fv;
hold on;

nii = load_nii('IIT_GM_Desikan_atlas.nii.gz');
PIB_ROI = double(nii.img);
cm = colormap('jet');
roi16_idx = unique(PIB_ROI);

for t=1:length(T_interp)
    close all;
    hold on;

    axis image;
    view([0 0 1]);
    shading interp
    for i=1:length(roi16_idx)
        v = (PIB_ROI == roi16_idx(i));
        
        % This if statement is just validating the mapping.
        if iit_to_tadpole_idx(i) ~= -1
            v = smooth3(v, 'gaussian', 11, 2);
            fv = isosurface(v);
            glass_pib.v = v;
            glass_pib.fv = fv;

            hold on;
            p = patch(glass_pib.fv);

            % Using the interpolated PiB measure that we precomputed.
            av45_T = av45_interp(iit_to_tadpole_idx(i),t);

            % This is where you set the ROI color intensity
            p.FaceColor = cm(round(av45_T*63)+1, :);
            p.EdgeColor = 'none';
            p.SpecularStrength = 0.4;
            p.DiffuseStrength = 0.6;
            p.AmbientStrength = 0.3;
            hold off;
        end
    end
    camlight(45, 45);
    camlight(-45, -45);
    axis off;


    % Top View
    view([-90 90])
    title(['t = ', sprintf('%.2f', T_interp(t))], 'FontSize', 20);
    export_fig(['ours_video_adas13/av45_adas13_top_', sprintf('%.2f', T_interp(t)), '.png'], '-transparent', '-r100');
    
    % Bottom View
    view([1 0 0])
    title(['t = ', sprintf('%.2f', T_interp(t))], 'FontSize', 20);
    export_fig(['ours_video_adas13/av45_adas13_front_', sprintf('%.2f', T_interp(t)), '.png'], '-transparent', '-r100');
     
end

%% This is identical to the loop above, except it only uses the ROIS of the
% right hemisphere by checking iit_to_tadpole_idx(i) < 42.
for t=1:length(T_interp)
    close all;
    hold on;

    axis image;
    view([0 0 1]);
    shading interp
    for i=1:length(roi16_idx)
        v = (PIB_ROI == roi16_idx(i));
        if iit_to_tadpole_idx(i) ~= -1 && iit_to_tadpole_idx(i) < 42
            v = smooth3(v, 'gaussian', 11, 2);
            fv = isosurface(v);
            glass_pib.v = v;
            glass_pib.fv = fv;

            hold on;
            p = patch(glass_pib.fv);

            av45_T = av45_interp(iit_to_tadpole_idx(i),t);

            p.FaceColor = cm(round(av45_T*63)+1, :);
            p.EdgeColor = 'none';
            p.SpecularStrength = 0.4;
            p.DiffuseStrength = 0.6;
            p.AmbientStrength = 0.3;
            hold off;
        end
    end
    camlight(45, 45);
    camlight(-45, -45);
    axis off;

    view([0 1 0])
    title(['t = ', sprintf('%.2f', T_interp(t))], 'FontSize', 20);
    export_fig(['ours_video_adas13/av45_adas13_side_', sprintf('%.2f', T_interp(t)), '.png'], '-transparent', '-r100');
    
end
