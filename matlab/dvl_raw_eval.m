data = readtable('/home/kraft/workspace/dvl-nav/notebook/dataframe.CSV');

N = length(data.abs_vel_btm_u);

vel_diff = zeros(N, 6, 3);
t = 1:1:N;

vel_diff(:,1,1) = data.abs_vel_btm_u + data.vel_bin0_beam0;
vel_diff(:,1,2) = data.abs_vel_btm_v + data.vel_bin0_beam1;
vel_diff(:,1,3) = data.abs_vel_btm_w + data.vel_bin0_beam2;
vel_diff(:,2,1) = data.abs_vel_btm_u + data.vel_bin1_beam0;
vel_diff(:,2,2) = data.abs_vel_btm_v + data.vel_bin1_beam1;
vel_diff(:,2,3) = data.abs_vel_btm_w + data.vel_bin1_beam2;
vel_diff(:,3,1) = data.abs_vel_btm_u + data.vel_bin2_beam0;
vel_diff(:,3,2) = data.abs_vel_btm_v + data.vel_bin2_beam1;
vel_diff(:,3,3) = data.abs_vel_btm_w + data.vel_bin2_beam2;
vel_diff(:,4,1) = data.abs_vel_btm_u + data.vel_bin3_beam0;
vel_diff(:,4,2) = data.abs_vel_btm_v + data.vel_bin3_beam1;
vel_diff(:,4,3) = data.abs_vel_btm_w + data.vel_bin3_beam2;
vel_diff(:,5,1) = data.abs_vel_btm_u + data.vel_bin4_beam0;
vel_diff(:,5,2) = data.abs_vel_btm_v + data.vel_bin4_beam1;
vel_diff(:,5,3) = data.abs_vel_btm_w + data.vel_bin4_beam2;
vel_diff(:,6,1) = data.abs_vel_btm_u + data.vel_bin5_beam0;
vel_diff(:,6,2) = data.abs_vel_btm_v + data.vel_bin5_beam1;
vel_diff(:,6,3) = data.abs_vel_btm_w + data.vel_bin5_beam2;

%%% Create shear matrix
dvl_vel_bins = zeros(N,15,3);
dvl_vel_bins(:,1,1) = -data.vel_bin0_beam0;
dvl_vel_bins(:,1,2) = -data.vel_bin0_beam1;
dvl_vel_bins(:,1,3) = -data.vel_bin0_beam2;
dvl_vel_bins(:,2,1) = -data.vel_bin1_beam0;
dvl_vel_bins(:,2,2) = -data.vel_bin1_beam1;
dvl_vel_bins(:,2,3) = -data.vel_bin1_beam2;
dvl_vel_bins(:,3,1) = -data.vel_bin2_beam0;
dvl_vel_bins(:,3,2) = -data.vel_bin2_beam1;
dvl_vel_bins(:,3,3) = -data.vel_bin2_beam2;
dvl_vel_bins(:,4,1) = -data.vel_bin3_beam0;
dvl_vel_bins(:,4,2) = -data.vel_bin3_beam1;
dvl_vel_bins(:,4,3) = -data.vel_bin3_beam2;
dvl_vel_bins(:,5,1) = -data.vel_bin4_beam0;
dvl_vel_bins(:,5,2) = -data.vel_bin4_beam1;
dvl_vel_bins(:,5,3) = -data.vel_bin4_beam2;
dvl_vel_bins(:,6,1) = -data.vel_bin5_beam0;
dvl_vel_bins(:,6,2) = -data.vel_bin5_beam1;
dvl_vel_bins(:,6,3) = -data.vel_bin5_beam2;
dvl_vel_bins(:,7,1) = -data.vel_bin6_beam0;
dvl_vel_bins(:,7,2) = -data.vel_bin6_beam1;
dvl_vel_bins(:,7,3) = -data.vel_bin6_beam2;
dvl_vel_bins(:,8,1) = -data.vel_bin7_beam0;
dvl_vel_bins(:,8,2) = -data.vel_bin7_beam1;
dvl_vel_bins(:,8,3) = -data.vel_bin7_beam2;
dvl_vel_bins(:,9,1) = -data.vel_bin8_beam0;
dvl_vel_bins(:,9,2) = -data.vel_bin8_beam1;
dvl_vel_bins(:,9,3) = -data.vel_bin8_beam2;
dvl_vel_bins(:,10,1) = -data.vel_bin9_beam0;
dvl_vel_bins(:,10,2) = -data.vel_bin9_beam1;
dvl_vel_bins(:,10,3) = -data.vel_bin9_beam2;
dvl_vel_bins(:,11,1) = -data.vel_bin10_beam0;
dvl_vel_bins(:,11,2) = -data.vel_bin10_beam1;
dvl_vel_bins(:,11,3) = -data.vel_bin10_beam2;
dvl_vel_bins(:,12,1) = -data.vel_bin11_beam0;
dvl_vel_bins(:,12,2) = -data.vel_bin11_beam1;
dvl_vel_bins(:,12,3) = -data.vel_bin11_beam2;
dvl_vel_bins(:,13,1) = -data.vel_bin12_beam0;
dvl_vel_bins(:,13,2) = -data.vel_bin12_beam1;
dvl_vel_bins(:,13,3) = -data.vel_bin12_beam2;
dvl_vel_bins(:,14,1) = -data.vel_bin13_beam0;
dvl_vel_bins(:,14,2) = -data.vel_bin13_beam1;
dvl_vel_bins(:,14,3) = -data.vel_bin13_beam2;
dvl_vel_bins(:,15,1) = -data.vel_bin14_beam0;
dvl_vel_bins(:,15,2) = -data.vel_bin14_beam1;
dvl_vel_bins(:,15,3) = -data.vel_bin14_beam2;

bin = 1;

% filter by magnitude of shear
% mag = sqrt(sum(vel_diff(:,bin,:).^2,3));
% [RowNrs,~] = find(mag>1.0);
% zmat = NaN(length(RowNrs),3);
% vel_diff(RowNrs,bin,:) = zmat;

% filter by pitch change
% [RowNrs,~] = find(abs(data.delta_pitch)>0.5);
% zmat = NaN(length(RowNrs),3);
% vel_diff(RowNrs,bin,:) = zmat;

% filter by magnitude of vehicle velocity
% mag = sqrt(data.abs_vel_btm_u.^2 + data.abs_vel_btm_v.^2);
% [RowNrs,~] = find(mag<0.2);
% zmat = NaN(length(RowNrs),3);
% vel_diff(RowNrs,bin,:) = zmat;

% average wc bin 0 velocities
s = 10;
vel_bin0_avg = zeros(N, 3);
vel_bin0_avg(:,1) = smoothdata(data.vel_bin0_beam0,'movmean',s);
vel_bin0_avg(:,2) = smoothdata(data.vel_bin0_beam1,'movmean',s);
vel_bin0_avg(:,3) = smoothdata(data.vel_bin0_beam2,'movmean',s);

vel_bin1_avg = zeros(N, 3);
vel_bin1_avg(:,1) = smoothdata(data.vel_bin1_beam0,'movmean',s);
vel_bin1_avg(:,2) = smoothdata(data.vel_bin1_beam1,'movmean',s);
vel_bin1_avg(:,3) = smoothdata(data.vel_bin1_beam2,'movmean',s);

vel_bin2_avg = zeros(N, 3);
vel_bin2_avg(:,1) = smoothdata(data.vel_bin2_beam0,'movmean',s);
vel_bin2_avg(:,2) = smoothdata(data.vel_bin2_beam1,'movmean',s);
vel_bin2_avg(:,3) = smoothdata(data.vel_bin2_beam2,'movmean',s);

vel_bin3_avg = zeros(N, 3);
vel_bin3_avg(:,1) = smoothdata(data.vel_bin3_beam0,'movmean',s);
vel_bin3_avg(:,2) = smoothdata(data.vel_bin3_beam1,'movmean',s);
vel_bin3_avg(:,3) = smoothdata(data.vel_bin3_beam2,'movmean',s);

vel_bin_avgs = zeros(N,4,3);
vel_bin_avgs(:,1,:) = vel_bin0_avg;
vel_bin_avgs(:,2,:) = vel_bin1_avg;
vel_bin_avgs(:,3,:) = vel_bin2_avg;
vel_bin_avgs(:,4,:) = vel_bin3_avg;

% note diff is this way to measure current relative to vehicle VTW
vel_bin_diff_avg = -vel_bin0_avg + vel_bin1_avg;

vel_bin_diff_avgs = zeros(N,3,3);
vel_bin_diff_avgs(:,1,:) = -vel_bin0_avg + vel_bin1_avg;
vel_bin_diff_avgs(:,2,:) = -vel_bin1_avg + vel_bin2_avg;
vel_bin_diff_avgs(:,3,:) = -vel_bin2_avg + vel_bin3_avg;

vel_voc_avg = zeros(N, 3);
vel_voc_avg(:,1) = data.abs_vel_btm_u + vel_bin0_avg(:,1);
vel_voc_avg(:,2) = data.abs_vel_btm_v + vel_bin0_avg(:,2);
vel_voc_avg(:,3) = data.abs_vel_btm_w + vel_bin0_avg(:,3);
% filter by magnitude of shear
% mag = sqrt(sum(vel_voc_avg(:,:).^2,2));
% [RowNrs,~] = find(mag>0.1);
% zmat = zeros(length(RowNrs),3);
% vel_voc_avg(RowNrs,:) = zmat;

vel_diff_avg = zeros(N,3);
vel_diff_avg(:,1) = smoothdata(vel_diff(:,1,1),'movmean',s);
vel_diff_avg(:,2) = smoothdata(vel_diff(:,1,2),'movmean',s);
vel_diff_avg(:,3) = smoothdata(vel_diff(:,1,3),'movmean',s);

tl = 45*60;

%%% Water Column
max_depth = 30;
wc_bin_size = 2;

% from diff between VOG and bin0
wc = zeros(floor(max_depth/2),3);
wc2 = zeros(floor(max_depth/2),3);
wc_count = zeros(floor(max_depth/2),1);
wc2_count = zeros(floor(max_depth/2),1);
for i = 1:tl
    wc_bin = floor(data.rel_pos_z(i)/wc_bin_size)+1;
    if ~anynan(vel_voc_avg(i,:))
        wc(wc_bin,:) = wc(wc_bin,:) + vel_voc_avg(i,:);
        wc_count(wc_bin) = wc_count(wc_bin) + 1;
    end
    if ~anynan(vel_bin_diff_avg(i,:))
        wc2(wc_bin+1,:) = wc2(wc_bin+1,:) + vel_bin_diff_avg(i,:);
        wc2_count(wc_bin+1) = wc2_count(wc_bin+1) + 1;
    end
end
wc = wc./wc_count;
wc2 = wc2./wc2_count;

% shear propagated
shear = zeros(floor(max_depth/2),3);
shear_count = zeros(floor(max_depth/2),1);
wcs = zeros(floor(max_depth/2),3);
wcs_count = zeros(floor(max_depth/2),1);
wcs_count(1) = 1;
wcs_count(2) = 1;
for i = 1:tl
    z = data.rel_pos_z(i);
%     pitch = data.ahrs_pitch(i) * pi/180;
%     roll = data.ahrs_roll(i) * pi/180;
    z_bin = floor(z/wc_bin_size)+1;
%     vtw = -vel_bin0_avg(i,:);
%     voc = wcs(z_bin,:)/wcs_count(z_bin);
%     vog = vtw + voc;
    good_bins = data.num_good_vel_bins(i);
    good_bins = min(good_bins, 5);
    for k = 2:good_bins
%         s_bin = GetWCBin(z, roll, pitch, k, data);
        s_bin = z_bin + k;
%         if s_bin > 0
%             wcs(s_bin,:) = wcs(s_bin,:) + vog - [dvl_vel_bins(i,k,1),dvl_vel_bins(i,k,2),dvl_vel_bins(i,k,3)];
%             wcs_count(s_bin) = wcs_count(s_bin) + 1;
%         end
%         if k == 1
%             shear(s_bin,:) = shear(s_bin,:) + [dvl_vel_bins(i,k,1),dvl_vel_bins(i,k,2),dvl_vel_bins(i,k,3)] - vtw;
%         else
%             shear(s_bin,:) = shear(s_bin,:) + [dvl_vel_bins(i,k,1),dvl_vel_bins(i,k,2),dvl_vel_bins(i,k,3)] - [dvl_vel_bins(i,k-1,1),dvl_vel_bins(i,k-1,2),dvl_vel_bins(i,k-1,3)];
%         end
        shear(s_bin,:) = shear(s_bin,:) + [dvl_vel_bins(i,k,1),dvl_vel_bins(i,k,2),dvl_vel_bins(i,k,3)] - [dvl_vel_bins(i,k-1,1),dvl_vel_bins(i,k-1,2),dvl_vel_bins(i,k-1,3)];
        shear_count(s_bin) = shear_count(s_bin) + 1;
    end
end
shear_accumulate = zeros(1,3);
for i = 1:length(shear_count)
    if shear_count(i) > 0
        shear(i,:) = shear(i,:)./shear_count(i);
    end
    shear_accumulate = shear_accumulate + shear(i,:);
    wcs(i,:) = shear_accumulate;
end
% wcs = wcs./wcs_count;
% wc_offset = sum(wc(3:5,:) - wcs(3:5,:))/3;
wc_offset = wc(4,:) - wcs(4,:);
wcs = wcs + wc_offset;

% bin 1-0 shear diff
shear2 = zeros(floor(max_depth/2),3);
shear_count2 = zeros(floor(max_depth/2),1);
wcs2 = zeros(floor(max_depth/2),3);
for i = 1:tl
    z = data.rel_pos_z(i);
    z_bin = floor(z/wc_bin_size)+1;
    s_bin = z_bin + 2;
    shear2(s_bin,:) = shear2(s_bin,:) + vel_bin_diff_avg(i,:);
    shear_count2(s_bin) = shear_count2(s_bin) + 1;
end
shear_accumulate = zeros(1,3);
for i = 1:length(shear_count2)
    if shear_count2(i) > 0
        shear2(i,:) = shear2(i,:)./shear_count2(i);
    end
    shear_accumulate = shear_accumulate + shear2(i,:);
    wcs2(i,:) = shear_accumulate;
end

wc_offset = sum(wc(3:5,:) - wcs2(3:5,:))./3;
wcs2 = wcs2 + wc_offset;

% wcs2(:,:) = zeros(size(wcs2));
% wcs2 = wcs2 + wc(4,:);

% bin 3-2-1-0 shear diff
shear3 = zeros(floor(max_depth/2),3);
shear_count3 = zeros(floor(max_depth/2),1);
wcs3 = zeros(floor(max_depth/2),3);
for i = 1:tl
    z = data.rel_pos_z(i);
    z_bin = floor(z/wc_bin_size)+1;
    for k = 1:2
        s_bin = z_bin + k + 1;
        shear3(s_bin,:) = shear3(s_bin,:) + squeeze(vel_bin_diff_avgs(i,k,:))';
        shear_count3(s_bin) = shear_count3(s_bin) + 1;
    end
end
shear_accumulate = zeros(1,3);
for i = 1:length(shear_count3)
    if shear_count3(i) > 0
        shear3(i,:) = shear3(i,:)./shear_count3(i);
    end
    shear_accumulate = shear_accumulate + shear3(i,:);
    wcs3(i,:) = shear_accumulate;
end

wc_offset = sum(wc(3:5,:) - wcs3(3:5,:))./3;
wcs3 = wcs3 + wc_offset;

%%% Odometery
odo = zeros(tl,3);
odo(1,:) = [0,0,0];

odo2 = zeros(tl,3);
odo2(1,:) = [0,0,0];

odo_raw = zeros(tl,3);
odo_raw(1,:) = [0,0,0];
odo_raw2 = zeros(tl,3);
odo_raw2(1,:) = [0,0,0];

pose = [0,0,0];
poser = [0,0,0];
pose2 = [0,0,0];
for i = 2:tl
    wc_bin = floor(data.rel_pos_z(i)/wc_bin_size)+1;
    dt = data.delta_t(i);
    deltaP = -vel_bin0_avg(i,:)*dt;
    deltaP1 = deltaP;
    deltaP1(1:2) = deltaP(1:2) + wc(wc_bin,1:2)*dt;
    if ~anynan(deltaP1)
        pose = pose + deltaP1;
    end
    odo(i,:) = pose;
    deltaP2 = deltaP;
    deltaP2(1:2) = deltaP(1:2) + wcs3(wc_bin,1:2)*dt;
%     deltaP2(1:2) = deltaP(1:2) + wcs2(wc_bin,1:2)*dt;
%     deltaP2(1:2) = deltaP(1:2) + wcs(wc_bin,1:2)*dt;
    if ~anynan(deltaP2)
        pose2 = pose2 + deltaP2;
    end
    odo2(i,:) = pose2;
    
%     deltaPr = -[data.vel_bin0_beam0(i), data.vel_bin0_beam1(i), data.vel_bin0_beam2(i)]*dt;
    deltaPr = -vel_bin0_avg(i,:)*dt;
    if ~anynan(deltaPr)
%     deltaPr = [data.rel_vel_dvl_u(i), data.rel_vel_dvl_v(i), data.rel_vel_dvl_w(i)]*dt;
        poser = poser + deltaPr;
    end
    odo_raw(i,:) = poser;
end

%%% Plots

figure(1);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),vel_diff(1:tl,1,1),vel_diff(1:tl,1,2),vel_diff(1:tl,1,3),'off');
title("voc bin 0")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(2);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),data.vel_bin0_beam0(1:tl),data.vel_bin0_beam1(1:tl),data.vel_bin0_beam2(1:tl),'off');
title("velocity bin 0")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(3);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),data.vel_bin0_beam0(1:tl)-data.vel_bin1_beam0(1:tl),data.vel_bin0_beam1(1:tl)-data.vel_bin1_beam1(1:tl),data.vel_bin0_beam2(1:tl)-data.vel_bin1_beam2(1:tl),'off');
title("diff bin 0 to bin 1")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(4);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),data.abs_vel_btm_u(1:tl), data.abs_vel_btm_v(1:tl), data.abs_vel_btm_w(1:tl),'off');
title("abs bottom vel")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(5);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),vel_bin0_avg(1:tl,1),vel_bin0_avg(1:tl,2),vel_bin0_avg(1:tl,3),'off');
title("smoothed velocity bin 0")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(6);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),vel_voc_avg(1:tl,1), vel_voc_avg(1:tl,2), vel_voc_avg(1:tl,3),'off');
title("diff abs to smoothed bin 0")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(7);
quiver3(zeros(tl,1),zeros(tl,1),-data.rel_pos_z(1:tl),vel_diff_avg(1:tl,1), vel_diff_avg(1:tl,2), vel_diff_avg(1:tl,3),'off');
title("voc bin 0 smoothed")
axis equal
xlim([-1 1])
ylim([-1 1])
zlim([-15 1])

figure(8);
% initialize position to FS estimate after vehicle dives to 4m from surface
sync_idx = 170;
odo_error_x = data.rel_pos_x(sync_idx) - odo(sync_idx,1);
odo_error_y = data.rel_pos_y(sync_idx) - odo(sync_idx,2);
odo(:,1) = odo(:,1) + odo_error_x;
odo(:,2) = odo(:,2) + odo_error_y;

odo2_error_x = data.rel_pos_x(sync_idx) - odo2(sync_idx,1);
odo2_error_y = data.rel_pos_y(sync_idx) - odo2(sync_idx,2);
odo2(:,1) = odo2(:,1) + odo2_error_x;
odo2(:,2) = odo2(:,2) + odo2_error_y;

plot3(odo(:,1), odo(:,2), -data.rel_pos_z(1:tl), 'green');
% % plot(odo(:,1), odo(:,2), 'green');
hold on;
% plot(odo2(:,1), odo2(:,2), 'black');
plot3(odo2(:,1), odo2(:,2), -data.rel_pos_z(1:tl), 'black');
plot3(data.rel_pos_x(1:tl), data.rel_pos_y(1:tl), -data.rel_pos_z(1:tl), 'blue')
% plot3(odo_raw(:,1), odo_raw(:,2), odo_raw(:,3), 'red')
plot3(odo_raw(:,1), odo_raw(:,2), -data.rel_pos_z(1:tl), 'red')
% plot(data.rel_pos_x_vtw(1:tl), data.rel_pos_y_vtw(1:tl), 'red')
hold off;
legend('VTW with WC from avg VTW', 'VTW w/ WC avg VoG-bin0 diff', 'VoG Odo', 'VTW from bin0 raw');

%%% Functions

function y = GetWCBin(z, r, p, n, data)
    D = [0, 0, data.bin0_distance(1) + (n-1)*data.depth_bin_length(1)];
    D = Qx(p) * Qy(r) * D';
    d = z + D(3);
    y = floor(d/data.depth_bin_length(1)) + 1;
end

function y = Qx(x)
    y = [1,       0,       0;
         0,  cos(x), -sin(x);
         0,  sin(x),  cos(x)];
end

function y = Qy(x)
    y = [cos(x),  0,  sin(x);
              0,  1,       0;
        -sin(x),  0,  cos(x)];
end

function y = Qz(x)
    y = [cos(x), -sin(x),  0;
         sin(x),  cos(x),  0;
              0,       0,  1;];
end