%% PiB ROI generation code for Zirui's MICCAI 2019
clear all
addpath('./altmany-export_fig-76bd7fa/');
%% FA Mask
nii = load_nii('CrispMeanIso_trMask.nii.gz');
nii_mask = nii;
v = double(nii.img);
[ndx, ndy, ndz] = ndgrid(1:(127/255):128, 1:(127/255):128, 1:(79/159):80);
v = interpn(v, ndx, ndy, ndz);
pad1 = zeros(256, 256, 28);
pad2 = zeros(256, 256, 68);
v = cat(3, pad1, v);
v = cat(3, v, pad2);
v = smooth3(v, 'gaussian', 11, 6);
fv = isosurface(v);

fv.vertices(:,[1 2]) = fv.vertices(:,[2 1]);
fv.vertices(:,1) = -fv.vertices(:,1);
% fv.vertices = 2*fv.vertices;
fv.vertices(:,3) = fv.vertices(:,3);

glass.v = v;
glass.fv = fv;

%% FA Mask
nii = load_nii('CrispMeanIso_fa.nii.gz');
v = double(nii.img);
[ndx, ndy, ndz] = ndgrid(1:(127/255):128, 1:(127/255):128, 1:(79/159):80);
v = interpn(v, ndx, ndy, ndz);
pad1 = zeros(256, 256, 28);
pad2 = zeros(256, 256, 68);
v = cat(3, pad1, v);
v = cat(3, v, pad2);
% v = smooth3(v, 'gaussian', 7, 3);


nii = load_nii('CrispMeanIso_trMask.nii.gz');
nii_mask = nii;
v_mask = double(nii.img);
[ndx, ndy, ndz] = ndgrid(1:(127/255):128, 1:(127/255):128, 1:(79/159):80);
v_mask = interpn(v_mask, ndx, ndy, ndz);
pad1 = zeros(256, 256, 28);
pad2 = zeros(256, 256, 68);
v_mask = cat(3, pad1, v_mask);
v_mask = cat(3, v_mask, pad2);
v_mask_orig = v_mask;
v_mask = smooth3(v_mask, 'gaussian', 7, 3);
% v(v_mask < 0.8) = max(v(:));
v(v_mask < 0.8) = 1;
v = smooth3(v, 'gaussian', 3, 1);
v_fa = v;

zmin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 1), 2)) > 0))
zmax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 1), 2)) > 0))
ymin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 1), 3)) > 0))
ymax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 1), 3)) > 0))
xmin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 2), 3)) > 0))
xmax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 2), 3)) > 0))
v_fa = v_fa(xmin_fa:xmax_fa, ymin_fa:ymax_fa, zmin_fa:zmax_fa);
[x_fa, y_fa, z_fa] = meshgrid(1:ymax_fa-ymin_fa+1, 1:xmax_fa-xmin_fa+1, 1:zmax_fa-zmin_fa+1);

save v_fa v_fa x_fa y_fa z_fa;

%%
load glass_fv;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1 = top, 2 = front, 3 = left
cam_view = 3;

% 1 = fiber bundles, 2 = pib rois
blob = 2;


%
close all;
pause(0.3);
figure(1);

load v_fa;
hold on;

zmin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 1), 2)) > 0))
zmax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 1), 2)) > 0))
ymin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 1), 3)) > 0))
ymax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 1), 3)) > 0))
xmin_fa = min(find(squeeze(sum(sum(v_mask > 0.8, 2), 3)) > 0))
xmax_fa = max(find(squeeze(sum(sum(v_mask > 0.8, 2), 3)) > 0))

[x_fa, y_fa, z_fa] = meshgrid(1:ymax_fa-ymin_fa+1, 1:xmax_fa-xmin_fa+1, 1:zmax_fa-zmin_fa+1);
xslice0 = (ymax_fa - ymin_fa) / 2 - 23;
yslice0 = (xmax_fa - xmin_fa + 1) / 2 - 10;
zslice0 = (zmax_fa - zmin_fa) / 2 + 10;
if cam_view == 1
    xslice = []; 
    yslice = []; 
    zslice = zslice0;
elseif cam_view == 2
    xslice = xslice0; 
    yslice = []; 
    zslice = [];
else
    xslice = []; 
    yslice = yslice0; 
    zslice = [];
end

% FA SLICE
freezeColors;
cm = [[0:64]' [0:64]' [0:64]'] / 64;
% cm = cm.^(0.7);
colormap(cm);
fa_slice = slice(x_fa, y_fa, z_fa, v_fa, xslice, yslice, zslice);
fa_slice_x = fa_slice.XData;
fa_slice.XData = -fa_slice.YData - (128 - yslice0 - 10);
fa_slice.YData = fa_slice_x;
fa_slice.YData = fa_slice.YData + (123 - xslice0 - 23);
fa_slice.ZData = fa_slice.ZData + (106 - zslice0 + 10);
fa_slice.FaceLighting = 'none';

if cam_view == 1
    fa_slice.ZData = fa_slice.ZData - 200;
elseif cam_view == 2
    fa_slice.YData = fa_slice.YData + 200;
else
    fa_slice.XData = fa_slice.XData + 200;
end
set(findobj(gca,'Type','Surface'),'EdgeColor','none');
% unfreezeColors;
% hold off;

if blob == 1
    freezeColors;
    % hold on;
    %         colormap('parula');
    %         colormap('jet');
    h = surf(x, y, z, c);
    h.EdgeColor = 'none';
    h.EdgeLighting = 'none';
    h.EdgeAlpha = 0;
    axis equal;
    % view([0 0 1]);
    camlight(45, 45);
    camlight(-45, -45);
    % view([0 0 1]);
    shading interp
    colormap('hsv');
    % unfreezeColors;
elseif blob == 2

    % NMS_NP vs. pTau
%     nms = [0.314, 0.350, 0.360, 0.420, 0.517, 0.441, 0.524, 0.269, 0.462, 0.323, 0.097, 0.462, 0.349, 0.349, 0.391, 0.409];
    % NMS_NP vs. hTau
%     nms = [0.397, 0.374, 0.444, 0.5, 0.514, 0.415, 0.502, 0.410, 0.504, 0.442, 0.141, 0.476, 0.38, 0.398, 0.351, 0.475];
    % NMS_GP vs. hTau
    nms = [0.9252, 1.0312, 1.0611, 1.0721, 1.1506, 1.1343, 0.9468, 0.9993, 0.9973, 0.9923, 0.9707, 1.0225, 1.1205, 1.0612, 0.8411, 0.9997]
    %nms =    [0.9295, 1.0377, 1.0647, 1.0787, 1.1499, 1.1393, 0.9453, 0.9974, 0.9943,0.9904, 0.9682, 1.0233, 1.1209, 1.0576, 0.8483, 1.0025]
    %nms =    [0.9335, 1.0439, 1.0682, 1.0849, 1.1494, 1.1440, 0.9439, 0.9958, 0.9916,0.9888, 0.9659, 1.0242, 1.1214, 1.0545, 0.8550, 1.0053]
    %nms =    [0.9371, 1.0498, 1.0716, 1.0909, 1.1490, 1.1486, 0.9427, 0.9944, 0.9891,0.9874, 0.9639, 1.0251, 1.1220, 1.0519, 0.8613, 1.0079]
    %nms =    [0.9404, 1.0554, 1.0748, 1.0965, 1.1487, 1.1531, 0.9418, 0.9932, 0.9868,0.9861, 0.9622, 1.0261, 1.1227, 1.0496, 0.8673, 1.0106]
%  nms =    [0.9435, 1.0606, 1.0780, 1.1019, 1.1485, 1.1574, 0.9409, 0.9922, 0.9848,0.9851, 0.9608, 1.0271, 1.1235, 1.0478, 0.8729, 1.0131]
    %nms =    [0.9464, 1.0657, 1.0810, 1.1070, 1.1484, 1.1615, 0.9403, 0.9915, 0.9830,0.9841, 0.9596, 1.0282, 1.1243, 1.0462, 0.8782, 1.0156]
    %nms =    [0.9491, 1.0705, 1.0839, 1.1119, 1.1485, 1.1655, 0.9398, 0.9908, 0.9814,0.9834, 0.9585, 1.0294, 1.1252, 1.0449, 0.8833, 1.0181]
    %nms =    [0.9516, 1.0750, 1.0868, 1.1166, 1.1486, 1.1694, 0.9394, 0.9904, 0.9799,0.9827, 0.9577, 1.0305, 1.1262, 1.0439, 0.8881, 1.0205]
    %nms =    [0.9540, 1.0794, 1.0896, 1.1211, 1.1488, 1.1731, 0.9391, 0.9901, 0.9786,0.9822, 0.9570, 1.0317, 1.1272, 1.0432, 0.8926, 1.0228]
 %   nms =    [0.9563, 1.0836, 1.0922, 1.1254, 1.1491, 1.1768, 0.9390, 0.9899, 0.9775, 0.9818, 0.9566, 1.0330, 1.1283, 1.0426, 0.8969, 1.0252]
        
    % PIB ROIS
    freezeColors;
    axis image;
    camlight(45, 45);
    camlight(-45, -45);
    % view([0 0 1]);
    shading interp
%     cm = colormap('parula');
%     cm = colormap('hsv');
    cm = colormap(jet(256));
%     load pib_fv_0.95_0.85_g11_sig2.mat
    load pib_fv_0.95_0.80_g11_sig2.mat
    chunck_1 = 256/(1.2 - 0.84);
    chunck_2 = 6/(1.2 - 0.84);
    for i=1:16
    %    hold on;
   %     p = patch(pib_fv(i).fv);
     %   p.FaceColor = cm(round( (nms(i) - 0.83)*chunck) , :);
%        p.FaceColor = cm(round( (nms(i)/0.55)*64), :);
     %   p.EdgeColor = 'none';
      %  hold off;
      p_xyz = mean(pib_fv(i).fv.vertices, 1);
       
        [x, y, z] = sphere;
        %radius = 10*nms(i);
        radius = round( (nms(i) - 0.7)*chunck_2);

        % radius is centered at 0 first, so multiplication makes it bigger
        x = x*radius;
        y = y*radius;
        z = z*radius;
       
        h = surf(x+p_xyz(1),y+p_xyz(2),z+p_xyz(3), 'FaceLighting', 'phong');
        h.FaceColor = cm(round( (nms(i) - 0.84)*chunck_1), :);
        h.EdgeColor = 'none';
        

    end
end
hcb=colorbar;
cb = hcb.Limits;


% GLASS FA MASK
hold on;
p = patch(glass_fv);
% isonormals(glass.v, p)
p.FaceColor = [0.85 0.85 0.85];
p.EdgeColor = 'none';
p.FaceLighting = 'none';
p.FaceVertexCData = [zeros(size(p.Vertices,1),1), zeros(size(p.Vertices,1),1), zeros(size(p.Vertices,1),1)];
% p.FaceAlpha = 0.2;
p.FaceAlpha = 0.2;
axis image;
view([0 0 1]);
hold off;

axis off;
fig = gcf;
fig.PaperUnits = 'inches';
set(gca,'position',[0 0 1 1],'units','normalized')
set(hcb,'YTick',linspace(0.2,0.98, 5));
set(hcb,'YTickLabel', cellstr(num2str(linspace(0.84,1.2, 5)', '%.2f')) )
hcb;

if cam_view == 1
    view([0 0 1])
    %camzoom(0.90);
    if blob == 1
        export_fig(['./figs_pib_wacs_5/fb_fa_top.png'], '-transparent', '-r600');
    elseif blob == 2
        export_fig(['./image/fb_pib_top.png'], '-transparent', '-r600');
    end
    set(gca,'position',[0 0 1 1],'units','normalized')
elseif cam_view == 2
    view([0 -1 0.001])
%     camzoom(0.90);
    if blob == 1
        export_fig(['./figs_pib_wacs_5/fb_fa_front.png'], '-transparent', '-r600');
    elseif blob == 2
        export_fig(['./image/fb_pib_front.png'], '-transparent', '-r600');
    end
    set(gca,'position',[0 0 1 1],'units','normalized')
else
    view([-1 0 0])
%     camzoom(0.90);
    if blob == 1
        export_fig(['./figs_pib_wacs_5/fb_fa_left.png'], '-transparent', '-r600');
    elseif blob == 2
        export_fig(['./image/fb_pib_left.png'], '-transparent', '-r600');
    end
    set(gca,'position',[0 0 1 1],'units','normalized')
end
%%
addpath('./altmany-export_fig-76bd7fa/');
view([0 0 1])
export_fig(['./figs_pib_wacs_2/fb_all_top.png'], '-transparent', '-r600');
set(gca,'position',[0 0 1 1],'units','normalized')
view([0 -1 0.001])
export_fig(['./figs_pib_wacs_2/fb_all_front.png'], '-transparent', '-r600');
set(gca,'position',[0 0 1 1],'units','normalized')
view([-1 0 0])
export_fig(['./figs_pib_wacs_2/fb_all_left.png'], '-transparent', '-r600');






