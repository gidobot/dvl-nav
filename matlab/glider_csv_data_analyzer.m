if exist('dbd_raw','var') == 0
    opts = detectImportOptions('/home/kraft/workspace/data/glider/PuertoRico/17MAR_TS1/DBD_MAR17.csv');
    dbd_raw = readtable("/home/kraft/workspace/data/glider/PuertoRico/17MAR_TS1/DBD_MAR17.csv",opts);
end
if exist('ahrs_raw','var') == 0
    opts = detectImportOptions('/home/kraft/workspace/data/glider/PuertoRico/17MAR_TS1/rosbag_output/devices-spartonm2-ahrs.csv');
    ahrs_raw = readtable("/home/kraft/workspace/data/glider/PuertoRico/17MAR_TS1/rosbag_output/devices-spartonm2-ahrs.csv",opts);
end

time_min = 1647520348;
time_max = 1647528872;
time_offset = 25;

ahrs_times = ahrs_raw.Time;
ahrs_heads = ahrs_raw.compass_heading;
dbd_times = dbd_raw.m_present_time + time_offset;
dbd_heads = (dbd_raw.m_heading * 180/pi);

ahrs_inds = (ahrs_times > time_min) & (ahrs_times < time_max);
ahrs_times = ahrs_times(ahrs_inds);
ahrs_heads = ahrs_heads(ahrs_inds);

dbd_inds = (dbd_times > time_min) & (dbd_times < time_max);
dbd_inds_logical = find(dbd_inds);
dbd_inds_min = min(dbd_inds_logical) - 1;
if dbd_inds_min < 1
    dbd_inds_min = 1;
end
dbd_inds_max = max(dbd_inds_logical) + 1;
if dbd_inds_max > length(dbd_times)
    dbd_inds_max = length(dbd_times);
end
dbd_times = dbd_times(dbd_inds_min:dbd_inds_max);
dbd_heads = dbd_heads(dbd_inds_min:dbd_inds_max);

dbd_heads_interpelated = zeros(size(ahrs_times));
i_dbd = 1;

for i = 1:length(ahrs_times)
    ref_time = ahrs_times(i);
    while dbd_times(i_dbd) < ref_time
        i_dbd = i_dbd+1;
    end
    if i_dbd > 1
        head_diff = dbd_heads(i_dbd) - dbd_heads(i_dbd-1);
        interp_frac = (ref_time - dbd_times(i_dbd-1)) / (dbd_times(i_dbd) - dbd_times(i_dbd-1));
        dbd_heads_interpelated(i) = dbd_heads(i_dbd-1) + (interp_frac*head_diff);
    end
end

heading_error = angdiff(dbd_heads_interpelated*pi/180, ahrs_heads*pi/180)*180/pi;

p1 = -1.93e-05; p2 = 0.01139; p3 = -1.647; p4 = 9.672;
dh = dbd_heads_interpelated;
heading_adj = p1*dh.^3 + p2*dh.^2 + p3*dh + p4;
dbd_heads_interpelated_adj = dbd_heads_interpelated + heading_adj;

figure(1);
hold off;
plot(ahrs_times-ahrs_times(1), dbd_heads_interpelated);
hold on;
plot(ahrs_times-ahrs_times(1), ahrs_heads);
title("Heading");
legend("frontseat", "backseat");
figure(2);
plot(ahrs_times-ahrs_times(1), heading_error);
title("Error");
trimmean(abs(heading_error), 10)
figure(3);
polarscatter(ahrs_heads*pi/180,heading_error);
figure(4);
title("Heading vs Error");
scatter(ahrs_heads*pi/180, heading_error);
figure(5);
hold off;
plot(ahrs_times-ahrs_times(1), dbd_heads_interpelated_adj);
hold on;
plot(ahrs_times-ahrs_times(1), ahrs_heads);
title("Heading Adjusted Frontseat");
legend("frontseat", "backseat");