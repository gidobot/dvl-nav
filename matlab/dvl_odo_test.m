data = readtable('/home/kraft/workspace/dvl-nav/notebook/dataframe.CSV');
data_fs = readtable('/home/kraft/workspace/dvl-nav/notebook/fs_dataframe.CSV');

N = length(data.abs_vel_btm_u);

vel_diff = zeros(N, 6, 3);
t = 1:N;

s = 5;  % bin0 vtw smoothing window

sample_number = 60; % number of WC shear measurements in matrix

%% Create shear matrix
dvl_vel_bins = zeros(N,15,3);
dvl_vel_bins(:,1,1) = -data.vel_bin0_beam0;
dvl_vel_bins(:,1,2) = -data.vel_bin0_beam1;
dvl_vel_bins(:,1,3) = -data.vel_bin0_beam2;
dvl_vel_bins(:,1,4) = -data.vel_bin0_beam3;
dvl_vel_bins(:,2,1) = -data.vel_bin1_beam0;
dvl_vel_bins(:,2,2) = -data.vel_bin1_beam1;
dvl_vel_bins(:,2,3) = -data.vel_bin1_beam2;
dvl_vel_bins(:,2,4) = -data.vel_bin1_beam3;
dvl_vel_bins(:,3,1) = -data.vel_bin2_beam0;
dvl_vel_bins(:,3,2) = -data.vel_bin2_beam1;
dvl_vel_bins(:,3,3) = -data.vel_bin2_beam2;
dvl_vel_bins(:,3,4) = -data.vel_bin2_beam3;
dvl_vel_bins(:,4,1) = -data.vel_bin3_beam0;
dvl_vel_bins(:,4,2) = -data.vel_bin3_beam1;
dvl_vel_bins(:,4,3) = -data.vel_bin3_beam2;
dvl_vel_bins(:,4,4) = -data.vel_bin3_beam3;
dvl_vel_bins(:,5,1) = -data.vel_bin4_beam0;
dvl_vel_bins(:,5,2) = -data.vel_bin4_beam1;
dvl_vel_bins(:,5,3) = -data.vel_bin4_beam2;
dvl_vel_bins(:,5,4) = -data.vel_bin4_beam3;
dvl_vel_bins(:,6,1) = -data.vel_bin5_beam0;
dvl_vel_bins(:,6,2) = -data.vel_bin5_beam1;
dvl_vel_bins(:,6,3) = -data.vel_bin5_beam2;
dvl_vel_bins(:,6,4) = -data.vel_bin5_beam3;
dvl_vel_bins(:,7,1) = -data.vel_bin6_beam0;
dvl_vel_bins(:,7,2) = -data.vel_bin6_beam1;
dvl_vel_bins(:,7,3) = -data.vel_bin6_beam2;
dvl_vel_bins(:,7,4) = -data.vel_bin6_beam3;
dvl_vel_bins(:,8,1) = -data.vel_bin7_beam0;
dvl_vel_bins(:,8,2) = -data.vel_bin7_beam1;
dvl_vel_bins(:,8,3) = -data.vel_bin7_beam2;
dvl_vel_bins(:,8,4) = -data.vel_bin7_beam3;
dvl_vel_bins(:,9,1) = -data.vel_bin8_beam0;
dvl_vel_bins(:,9,2) = -data.vel_bin8_beam1;
dvl_vel_bins(:,9,3) = -data.vel_bin8_beam2;
dvl_vel_bins(:,9,4) = -data.vel_bin8_beam3;
dvl_vel_bins(:,10,1) = -data.vel_bin9_beam0;
dvl_vel_bins(:,10,2) = -data.vel_bin9_beam1;
dvl_vel_bins(:,10,3) = -data.vel_bin9_beam2;
dvl_vel_bins(:,10,4) = -data.vel_bin9_beam3;
dvl_vel_bins(:,11,1) = -data.vel_bin10_beam0;
dvl_vel_bins(:,11,2) = -data.vel_bin10_beam1;
dvl_vel_bins(:,11,3) = -data.vel_bin10_beam2;
dvl_vel_bins(:,11,4) = -data.vel_bin10_beam3;
dvl_vel_bins(:,12,1) = -data.vel_bin11_beam0;
dvl_vel_bins(:,12,2) = -data.vel_bin11_beam1;
dvl_vel_bins(:,12,3) = -data.vel_bin11_beam2;
dvl_vel_bins(:,12,4) = -data.vel_bin11_beam3;
dvl_vel_bins(:,13,1) = -data.vel_bin12_beam0;
dvl_vel_bins(:,13,2) = -data.vel_bin12_beam1;
dvl_vel_bins(:,13,3) = -data.vel_bin12_beam2;
dvl_vel_bins(:,13,4) = -data.vel_bin12_beam3;
dvl_vel_bins(:,14,1) = -data.vel_bin13_beam0;
dvl_vel_bins(:,14,2) = -data.vel_bin13_beam1;
dvl_vel_bins(:,14,3) = -data.vel_bin13_beam2;
dvl_vel_bins(:,14,4) = -data.vel_bin13_beam3;
dvl_vel_bins(:,15,1) = -data.vel_bin14_beam0;
dvl_vel_bins(:,15,2) = -data.vel_bin14_beam1;
dvl_vel_bins(:,15,3) = -data.vel_bin14_beam2;
dvl_vel_bins(:,15,4) = -data.vel_bin14_beam3;

%% average wc bin0 velocities
vel_bin0_avg = zeros(N, 3);
vel_bin0_avg(:,1) = smoothdata(data.vel_bin0_beam0,'movmedian',s);
vel_bin0_avg(:,2) = smoothdata(data.vel_bin0_beam1,'movmedian',s);
vel_bin0_avg(:,3) = smoothdata(data.vel_bin0_beam2,'movmedian',s);

%% Water Column Estimation
max_depth = 30; % in meters
wc_bin_size = 2; % divide water column into bins

% Ground truth current measurements from bottom lock and bin0 vtw
vel_voc_avg = zeros(N, 3);
vel_voc_avg(:,1) = data.abs_vel_btm_u + vel_bin0_avg(:,1);
vel_voc_avg(:,2) = data.abs_vel_btm_v + vel_bin0_avg(:,2);
vel_voc_avg(:,3) = data.abs_vel_btm_w + vel_bin0_avg(:,3);

%% Create time matrix for legs in dataset
T = [];
td = -1;
tx = -1;
ts = 1;
for i = 1:N
    if data.rel_pos_z(i) < 1 && ts == -1
        ts = i;
        td = -1;
        tx = -1;
    elseif data.rel_pos_z(i) > 1 && td == -1
        td = i;
    elseif data.rel_pos_z(i) > 4 && tx == -1
        tx = i;
        T = [T;[ts,td,tx]];
        ts = -1;
    end
end

%% Odometery variables
pose = [0,0,0];
pose_raw = [0,0,0];
pose_gt = [0,0,0];

odo_gt = zeros(N,3);
odo_gt(1,:) = [0,0,0];

odo_raw = zeros(N,3);
odo_raw(1,:) = [0,0,0];

odo = zeros(N,3);
odo(1,:) = [0,0,0];

sidx = 0;
fidx = 0;

%% Loop over legs
% for leg = 1:size(T,1)
for leg = 7:7

    ts = T(leg,1);
    if leg ~= size(T,1)
        tl = T(leg+1,1);
    else
        tl = N;
    end
    NN = tl-ts+1;
    td = T(leg,2);
    tx = T(leg,3);

    if sidx == 0
        sidx = ts;
        pose = [data.rel_pos_x(ts), data.rel_pos_y(ts), -data.rel_pos_z(ts)];
        pose_gt = pose;
        pose_raw = pose;
        odo(1:ts,:) = odo(1:ts,:) + pose;
        odo_gt(1:ts,:) = odo_gt(1:ts,:) + pose;
        odo_raw(1:ts,:) = odo_raw(1:ts,:) + pose;
    end

    % Ground truth WC from diff between VOG and bin0
    wc_gt = zeros(floor(max_depth/2),3);
    wc_gt_count = zeros(floor(max_depth/2),1);
    for i = ts:tl
        wc_bin = floor(data.rel_pos_z(i)/wc_bin_size)+1;
        if ~anynan(vel_voc_avg(i,:))
            wc_gt(wc_bin,:) = wc_gt(wc_bin,:) + vel_voc_avg(i,:);
            wc_gt_count(wc_bin) = wc_gt_count(wc_bin) + 1;
        end
    end
    wc_gt = wc_gt./wc_gt_count;

    % WC from surface bottom lock and shears
%     fs_ts = 2400;
%     fs_tl = 2500;
%     fgps = [0,0,0];
%     ft   = 0;
%     sgps = [0,0,0];
%     st   = 0;
%     for i = fs_ts:fs_tl
%         if ft == 0 && ~isnan(data_fs.m_gps_x_lmc(i))
%             ft = data_fs.time(i);
%             [e, n, z] = get_utm_coords_from_glider_lat_lon(data_fs.m_gps_lat(i), data_fs.m_gps_lon(i));
%             fgps = [e, n, 0];
%         end
%         if ~isnan(data_fs.m_gps_x_lmc(i))
%             st = data_fs.time(i);
%             [e, n, z] = get_utm_coords_from_glider_lat_lon(data_fs.m_gps_lat(i), data_fs.m_gps_lon(i));
%             sgps = [e, n, 0];
%         end
%     end
%     pds = sgps - fgps;
%     vds = pds./(st-ft);

    pds = [data.rel_pos_x(td), data.rel_pos_y(td), 0] - [data.rel_pos_x(ts), data.rel_pos_y(ts), 0];
    vds = pds./(td-ts);

    wc_gt2 = zeros(floor(max_depth/2),3);
    wc_gt2_count = zeros(floor(max_depth/2),1);
%     wc_gt2(1,:) = [median(data.abs_vel_btm_u(ts:td),'omitnan'), median(data.abs_vel_btm_v(ts:td),'omitnan'), median(data.abs_vel_btm_w(ts:td),'omitnan')];
    % wc_gt2(1,:) = [mean(data.abs_vel_btm_u(ts:tl2)), mean(data.abs_vel_btm_v(ts:tl2)), mean(data.abs_vel_btm_w(ts:tl2))];
    wc_gt2(1,:) = vds;
    wc_gt2(2:8,:) = -squeeze(median(dvl_vel_bins(ts:td,1:7,1:3),'omitnan')) + wc_gt2(1,:);

    %% Odometery from raw bin0 VTW and from bin0 VTW + Ground Truth WC
    
    % model based estimate of vtw does not work well for shallow angles
    % vz = data.delta_z./data.delta_t;
    % vh = zeros(size(vz));
    % idx = data.delta_z < 0;
    % vh(idx) = vz(idx)./tan((3+data.ahrs_pitch(idx)).*pi/180);
    % idx = data.delta_z > 0;
    % vh(idx) = vz(idx)./tan((-3+data.ahrs_pitch(idx)).*pi/180);
    
    for i = 2:NN
        idx = i+ts-1;
        wc_bin = floor(data.rel_pos_z(idx)/wc_bin_size)+1;
        dt = data.delta_t(idx);

        % odometery with GT WC
        deltaP = [0,0,0];
        if wc_bin ~= 1
            deltaP = -vel_bin0_avg(idx,:)*dt;
        end
        deltaP(1:2) = deltaP(1:2) + wc_gt(wc_bin,1:2)*dt;
        if ~anynan(deltaP)
            pose_gt = pose_gt + deltaP;
        end
        pose_gt(3) = -data.rel_pos_z(idx);
        if idx == tx
            pose_gt = [data.rel_pos_x(idx), data.rel_pos_y(idx), -data.rel_pos_z(idx)];
        end
        odo_gt(idx,:) = pose_gt;
        
        % odometery with only bin0 VTW
        deltaPr = -vel_bin0_avg(idx,:)*dt;
        if ~anynan(deltaPr)
            pose_raw = pose_raw + deltaPr;
        end
        if idx == tx
            pose_raw = [data.rel_pos_x(idx), data.rel_pos_y(idx), -data.rel_pos_z(idx)];
        end
        odo_raw(idx,:) = pose_raw;
    end

    %% Full odometery with iterative WC measurement
    wc = NaN(floor(max_depth/2),sample_number,3);
    for i = 1:sample_number
        wc(:,i,:) = wc_gt2;
    end

    voc_error = [0,0,0];
    btm_lock_counter = 0;
    dt_error = 0;
    first_pose_error = 1;
    for i = 2:NN
        idx = i+ts-1;
        z = data.rel_pos_z(idx);
        wc_bin = floor(z/wc_bin_size)+1;
        dt = data.delta_t(idx);
    
        % artificial bottom lock switch below 8 meters
        btm_lock_flag = 0;
        if z > 8
            btm_lock_flag = 1;
        end

        btm_lock_vog = [0,0,0];
        if btm_lock_flag
            btm_lock_vog = [data.abs_vel_btm_u(idx), data.abs_vel_btm_v(idx), data.abs_vel_btm_w(idx)];
        end
    
        voc = squeeze(median(wc(wc_bin,:,:),'omitnan'))';
        voc1 = squeeze(median(wc(wc_bin+1,:,:),'omitnan'))';

        vtw = [0,0,0];
        if wc_bin ~= 1
            voc_diff = voc1 - voc;
            vtw = -vel_bin0_avg(idx,:) + voc_diff;
    %         vtw = [dvl_vel_bins(idx,1,1),dvl_vel_bins(idx,1,2),dvl_vel_bins(idx,1,3)] + voc_diff;
        end
    
        vog = vtw + voc;
        
        % skip shear update when surface drifting
        if wc_bin > 1
            % accumulate new shear measurements
            good_bins = data.num_good_vel_bins(idx);
            for k = 2:good_bins
                s_bin = wc_bin + k;
                s_voc = vog - [dvl_vel_bins(idx,k,1),dvl_vel_bins(idx,k,2),dvl_vel_bins(idx,k,3)];
                s_voc = reshape(s_voc,[1,1,3]);
                wc(s_bin,:,:) = [wc(s_bin,2:end,:), s_voc];
            end
        end
        
        % estimate pose error update when there is bottom lock only if
        % range off of bottom is high enough
        btm_vel_error = abs(data.btm_beam3_velocity(idx));
        if btm_lock_flag
            if btm_vel_error < 0.02 && data.btm_beam0_range(idx) > 4
                btm_lock_counter = btm_lock_counter + 1;
                dv = btm_lock_vog - vog;
                voc_error = voc_error + (dv - voc_error)/btm_lock_counter;
            end
        else
            btm_lock_counter = 0;
            voc_error = [0,0,0];
        end
        
        % when bottom lock has been held long enough
        pose_error = [0,0,0];
        if btm_lock_counter > 30
            pose_error = voc_error * dt_error;
            % below assumes linear drift in velocity error from last
            % measurement, which gives the 0.5 factor
    %         if ~first_pose_error
    %             pose_error = 0.5*pose_error;
    %         else
    %             first_pose_error = 0;
    %         end
            
            voc_update = reshape(voc_error,[1,1,3]);
            wc = wc + voc_update;
    
            btm_lock_counter = 0;
            voc_error = [0,0,0];
            dt_error = 0;
        end
       
        deltaP = [0,0,0];
        if btm_lock_flag
            deltaP = btm_lock_vog*dt;
        else
            deltaP = vog*dt;
            dt_error = dt_error + dt;
        end
    
        if ~anynan(deltaP)
            pose = pose + deltaP;
        end
        if ~anynan(pose_error)
            pose = pose + pose_error;
        end
        % synchronize pose on dive depth
        if idx == tx
            pose = [data.rel_pos_x(idx), data.rel_pos_y(idx), -data.rel_pos_z(idx)];
            pose_error = [0,0,0];
            dt_error = 0;
        end
        odo(idx,:) = pose;
        
        % store idx range for plotting
        fidx = idx;
    end

end

%%% Plots

figure(1);

plot3(odo(sidx:fidx,1), odo(sidx:fidx,2), -data.rel_pos_z(sidx:fidx), 'green');
hold on;
plot3(odo_gt(sidx:fidx,1), odo_gt(sidx:fidx,2), -data.rel_pos_z(sidx:fidx), 'black');
plot3(data.rel_pos_x(sidx:fidx), data.rel_pos_y(sidx:fidx), -data.rel_pos_z(sidx:fidx), 'blue')
plot3(odo_raw(sidx:fidx,1), odo_raw(sidx:fidx,2), -data.rel_pos_z(sidx:fidx), 'red')

utm_offset = [data.rel_pos_x_utm(1), data.rel_pos_y_utm(1), 0];
time_zone_shift = data_fs.time(760) - data.time(2654);
data_fs.time = data_fs.time - time_zone_shift;
stime = data.time(sidx);
ftime = data.time(fidx);
for i = 1:length(data_fs.time)
    if data_fs.time(i) > stime && data_fs.time(i) < ftime
        [e, n, z] = get_utm_coords_from_glider_lat_lon(data_fs.m_gps_lat(i), data_fs.m_gps_lon(i));
        plot3(e-utm_offset(1), n-utm_offset(2), 0,'-o','Color','r','MarkerSize',10,'MarkerFaceColor','r');
    end
end

xlabel('x');
ylabel('y');
legend('Iterative Odo', 'Odo w/ GT WC', 'VoG Odo', 'VTW raw', 'GPS Fix');

hold off;

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

function [easting, northing, zone] = get_utm_coords_from_glider_lat_lon(m_lat, m_lon)
    if isnan(m_lat) || isnan(m_lon)
        easting = nan;
        northing = nan;
        zone = nan;
    else
        SECS_IN_MIN = 60;
        MIN_OFFSET = 100;
        lat_min  = fmod(m_lat, MIN_OFFSET);
        lon_min  = fmod(m_lon, MIN_OFFSET);
        lat_dec  = (m_lat - lat_min)/MIN_OFFSET + lat_min/SECS_IN_MIN;
        lon_dec  = (m_lon - lon_min)/MIN_OFFSET + lon_min/SECS_IN_MIN;
        [e, n, z]  = deg2utm(lat_dec, lon_dec);
        easting  = round(e,2);
        northing = round(n,2);
        zone     = z;
    end
end
 
function m = fmod(a, b)

    % Where the mod function returns a value in region [0, b), this
    % function returns a value in the region [-b, b), with a negative
    % value only when a is negative.

    if a == 0
        m = 0;
    else
        m = mod(a, b) + (b*(sign(a) - 1)/2);
    end

end
