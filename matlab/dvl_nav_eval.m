fname = 'data_dict.json';
fid = fopen(fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
data = jsondecode(str);

data.x = data.x - data.x(1);
data.y = data.y - data.y(1);

% plot3(data.x,data.y,-data.z);

clear x y z u v w su sv sw;

for i = 1:length(data.x)
    shear_list = data.shear(i);
    wc_list    = data.wc(i);
    for j = 1:length(shear_list{1})
        shear = shear_list{1}(j,:);
        wc = wc_list{1}(j,:);
        x(i+j-1) = data.x(i);
        y(i+j-1) = data.y(i);
        z(i+j-1) = -data.z(i)-2*j;
        su(i+j-1) = shear(1);
        sv(i+j-1) = shear(2);
        sw(i+j-1) = 0;
        u(i+j-1) = data.u(i);
        v(i+j-1) = data.v(i);
        w(i+j-1) = data.w(i);
        wcu(i+j-1) = wc(1);
        wcv(i+j-1) = wc(2);
        wcw(i+j-1) = 0;
%         p0(i,:) = [data.x(i), data.y(i), data.z(i)-2*j];
%         p1(i,:) = [data.x(i)+shear(1), data.y(i)+shear(2), data.z(i)-2*j];
    end
end

% arrow3(p0,p1);
figure(1);
quiver3(x,y,z,su,sv,sw,1);
title("shear")

figure(2);
quiver3(x,y,z,wcu,wcv,wcw,1);
title("wc")

figure(3);
quiver3(x,y,z,u,v,w,2);
title("vehicle velocity");