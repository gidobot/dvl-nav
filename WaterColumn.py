# WaterColumn.py
#
# Utility classes and functions for water column current estimation/ 
#   2022-07-22  gidobot@umich.edu

import numpy as np
from math import floor, ceil, fmod

class OceanCurrent(object):
    def __init__(self, u=None, v=None, w=None):
        """Represents a 3D OceanCurrent velocity."""
        self.u = u
        self.v = v
        self.w = w
        if not((u==None and v==None and w==None) or \
               (u!=None and v!=None and w!=None)):
            raise ValueError('bad ocean current',u,v,w)

    def __str__(self):
        if self.is_none():
            return 'V[----,----,----]'
        else:
            return 'V[%f,%f,%f]' % (self.u, self.v, self.w)

    def __eq__(self, other):
        return(str(self)==str(other))

    def copy(self):
        return OceanCurrent(self.u, self.v, self.w)

    def is_none(self):
        return((self.u==None     and self.v==None     and self.w==None) or 
               (np.isnan(self.u) and np.isnan(self.v)))

    def mag(self):
        if self.is_none():
            return np.NaN
        else:
            return np.linalg.norm([self.u,self.v,self.w])

    def subtract_shear(self, shear):
        """Subtract one OceanCurrent from another OceanCurrent."""
        new_shear_node = self.copy()
        if not new_shear_node.is_none():
            delta_u,delta_v,delta_w = shear.u,shear.v,shear.w
            new_shear_node.u += -delta_u
            new_shear_node.v += -delta_v
            new_shear_node.w += -delta_w
        return new_shear_node

    def add_shear(self, shear):
        """Add two OceanCurrent objects together."""
        new_shear_node = self.copy()
        if not new_shear_node.is_none():
            delta_u,delta_v,delta_w = shear.u,shear.v,shear.w
            new_shear_node.u += delta_u
            new_shear_node.v += delta_v
            new_shear_node.w += delta_w
        return new_shear_node

class Shear(object):
    def __init__(self, z_true, z_bin, t, shear_list=[],
        voc=OceanCurrent(), voc_ref=OceanCurrent(), voc_delta=OceanCurrent(),
        direction='descending', pitch=0, roll=0):
        """Represents a velocity shear measured at an instance in time from 
        the DVL. The Shear is the main data type that is managed by the 
        WaterColumn class. 

        Args:
            z_true: transducer depth in meters
            z_bin: water column bin in meters
            t: time in seconds
            water_col: water column object that Shear is a part of 
            shear_list: the velocity shears recorded by the DVL by comparing
                the dead-reckoned velocity and the DVL bin velocities.
            pitch: pitch of the glider in radians
            roll: roll of the glider in radians 
        """
        self.z_true    = z_true
        self.z_bin     = z_bin
        self.t         = t
        self.shear_list  = shear_list
        self.voc       = voc
        self.voc_ref   = voc_ref
        self.voc_delta = voc_delta
        self.direction = direction
        self.pitch     = pitch
        self.roll      = roll
        self.btm_track = False
        if not(direction!='descending' or direction!='ascending'):
            raise ValueError('bad direction value: %s' % direction)

    def __str__(self):
        # return string for shear node
        return('Shear<z:%3d, t:%4d, %s, %8s>' % 
               (self.z_bin, self.t, str(self.voc), voc_type))

    def has_voc(self):
        """returns true iff ocean velocity is currently specified

        If not yet specified, back propagation will be called later """
        return(not self.voc is None)

    def set_voc(self,val):
        """updates ocean current velocity"""
        self.voc=val.copy()

    def set_voc_delta(self,val):
        """updates ocean current velocity"""
        self.voc_delta=val.copy()

    def set_btm_track(self,boolean):
        """updates bottom track flag"""
        self.btm_track = boolean


class WaterColumn(object):
    def __init__(self,bin_len=2,bin0_dist=2.91,max_depth=1000,start_filter=0,end_filter=1,voc_mag_filter=1.0, voc_delta_mag_filter=0.20,
        voc_time_filter=10*60, sample_number=10):
        """Represents water column currents in an absolute reference frame.

        Uses measurements from Doppler Velocity Log (DVL) to determine water
        column velocities. The shear-based velocity method is used to 
        propagate water column currents forward and backward in time. Assumes
        downward facing DVL

        Args:
            bin_len: length of DVL depth bin.
            bin0_dist: distance from transducer head to middle of first bin.
            max_depth: max depth considered in the water column.
            start_filter: used to filter out the first number of DVL bins from 
                the propagation process.
            end_filter: used to filter out the last number of DVL bins from 
                the propagation process.
        """
        self._BIN_LEN      = bin_len
        self._BIN0_DIST    = bin0_dist
        self._MAX_DEPTH    = max_depth
        self._START_FILTER = start_filter
        self._END_FILTER   = end_filter
        self._WC_BIN_LEN   = int(bin_len)

        self.avg_voc_dict    = {i : OceanCurrent() for i in 
                               range(0,self.MAX_DEPTH,self.WC_BIN_LEN)}

        self.t_ref = 0

        # tuning parameters for ocean current estimation
        self.voc_mag_filter = voc_mag_filter
        self.voc_delta_mag_filter = voc_delta_mag_filter
        self.voc_time_filter = voc_time_filter  # time threshold (seconds) at which measurement becomes stale

        # wc matrix with each matrix element of the form (u, v, w, t)
        self.wc = np.zeros((int(self.MAX_DEPTH/self.WC_BIN_LEN), sample_number, 4))
        # set all times to -1 (invalid)
        self.wc[:,:,3] = -np.ones((self.wc.shape[0], sample_number))

    def __str__(self):
        string  = 'Water Column (depth=%0.f) \n' % (self.MAX_DEPTH)
        # for z in self.shear_node_dict.keys():
            # string += '|z =%3d|' % z 
            # for sn in self.shear_node_dict[z]:
                # string += ' '
                # string += str(sn)
            # string += '\n'
        return(string)

    @property
    def BIN_LEN(self):
        return self._BIN_LEN

    @property
    def BIN0_DIST(self):
        return self._BIN0_DIST

    @property
    def MAX_DEPTH(self):
        return self._MAX_DEPTH

    @property
    def START_FILTER(self):
        return self._START_FILTER

    @property
    def END_FILTER(self):
        return self._END_FILTER

    @property
    def WC_BIN_LEN(self):
        return self._WC_BIN_LEN

    def compute_averages(self):
        """Computes average water column currents for each depth bin."""
        # iterate over the depth bins 
        voc_u_list = []
        voc_v_list = []
        voc_w_list = []
        z_list     = []
        for z_bin in range(self.wc.shape[0]):
            V = np.zeros(3)
            count = 0
            voc = OceanCurrent()
            # TODO: make this a matrix operation
            for i in range(self.wc.shape[1]):
                if self.wc[z_bin, i, 3] >= 0:
                    V += self.wc[z_bin, i, 0:3]
                    count += 1
            if count > 0:
                V = V/count
                voc = OceanCurrent(V[0], V[1], V[2])
                voc_u_list.append(V[0])
                voc_v_list.append(V[1])
                voc_w_list.append(V[2])
                z_list.append(z_bin)
                self.avg_voc_dict[int(z_bin*self.WC_BIN_LEN)] = voc
            else:
                voc_u_list.append(np.NaN)
                voc_v_list.append(np.NaN)
                voc_w_list.append(np.NaN)
                z_list.append(z_bin)
        return (np.array(voc_u_list), 
                np.array(voc_v_list), 
                np.array(voc_w_list),
                np.array(z_list))
        
    def averages_to_str(self):
        """Converts averages to string format after they have been computed."""
        string  = 'Water Column (depth=%0.f) \n' % (self.MAX_DEPTH)
        for z_bin in range(self.wc.shape[0]):
            z = int(z_bin*self.WC_BIN_LEN)
            string += '|z =%3d| ' % z
            string += str(self.avg_voc_dict[z])
            string += '\n'
        return(string)

    def get_z_true(self, depth, roll, pitch, bin_num):
        """Get the true depth of the DVL depth bin

        Args: 
            parent: the parent node 
            bin_num: the DVL bin number removed from transducer (parent node)
        """
        # DEG_TO_RAD = np.pi/180
        # scale = np.cos(parent.pitch*DEG_TO_RAD)*np.cos(parent.roll*DEG_TO_RAD)
        # return (parent.z_true + self.BIN0_DIST+bin_num*self.BIN_LEN)*scale 
        D = np.array([0, 0, self.BIN0_DIST+(bin_num*self.BIN_LEN)])
        D = self.Qx(pitch) @ self.Qy(roll) @ D
        return depth + D[2]

    def get_wc_bin(self, z_true):
        """Get the depth of the water column cell."""
        z_bin = floor(z_true/self.WC_BIN_LEN)
        if z_bin < 0:
            z_bin = 0
        elif z_bin > self.wc.shape[0]-1:
            z_bin = self.wc.shape[0]-1
        return z_bin

    def mag_filter(self, voc, voc_delta):
        """Return true iff node meets magnitude reqs on voc and delta"""
        if not np.isnan(voc_delta.mag()):
            if voc_delta.mag() > self.voc_delta_mag_filter:
                return(False)
        if not np.isnan(voc.mag()):
            if voc.mag() > self.voc_mag_filter:
                return(False)
        return(True)

    def time_filter(self, t_new, t_ref):
        """Return true iff t is valid and recent within time threshold"""
        # -1 is used for invalid data points
        if (t_ref < 0) or (t_new - t_ref >= self.voc_time_filter):
            return False
        return True

    def get_voc_at_depth(self, z, t):
        """Get the water column currents recorded at a particular depth."""
        z_bin = self.get_wc_bin(z)
        V = np.zeros(3)
        count = 0
        voc = OceanCurrent()
        # TODO: make this a matrix operation
        # for j in range(max(0, z_bin-1), min(self.MAX_DEPTH,z_bin+1)):
        for i in range(self.wc.shape[1]):
            if self.time_filter(t, self.wc[z_bin, i, 3]):
                V += self.wc[z_bin, i, 0:3]
                count += 1
        if count:
            V = V/count
            voc = OceanCurrent(V[0], V[1], V[2])

        # TODO: Special case when data is stale but not invalid
        # Use simple average for now, but might be better to use robust filtering
        return voc, count


    def get_shear_bin_idx_from_depth(self, z_ref, rel_wc_bin, roll, pitch):
        if rel_wc_bin == 0:
            return None
        b0d = np.array([0, 0, self.BIN0_DIST])
        b0d = self.Qx(pitch) @ self.Qy(roll) @ b0d
        b0d = b0d[2]
        bl = np.array([0, 0, self.BIN_LEN])
        bl = self.Qx(pitch) @ self.Qy(roll) @ bl
        bl = bl[2]
        z_diff = rel_wc_bin*self.WC_BIN_LEN - z_ref%self.WC_BIN_LEN
        zr = z_diff - b0d
        if zr < 0:
            return -1
        bin_idx = floor(zr/bl)+1
        return bin_idx

    def find_voc_ref(self, z_ref, t, roll, pitch, num_bins):
        if num_bins < 1:
            voc, count = self.get_voc_at_depth(z_ref, t)
            return voc, None
        tb   = self.get_wc_bin(z_ref)
        z_bb = self.get_z_true(z_ref, roll, pitch, num_bins-1)
        bb   = self.get_wc_bin(z_bb)
        t_sub_wc = self.wc[tb:bb+1,:,3]
        t_sub_wc = np.where(t_sub_wc >= max(0,t-self.voc_time_filter), 1, 0)
        count = t_sub_wc.sum(1)
        rel_idx = count.argmax()
        z_bin = rel_idx + tb
        voc, count = self.get_voc_at_depth(z_bin*self.WC_BIN_LEN, t)
        bin_idx = self.get_shear_bin_idx_from_depth(z_ref, rel_idx, roll, pitch)
        return voc, bin_idx 

    def add_new_shear(self, z_true, t, shear_list, voc_ref=OceanCurrent(), 
        direction='descending', pitch=0, roll=0):
        """Adds a new DVL observation to the water column object.

        This is the main workhorse method for estimating ocean currents from
        DVL measurements: this function includes forward and backwards velocity
        shear propagations.
        """
        # TODO: currently returns 0 velocity if no prior measurement. support surface current estimate from gps,
        # but that should be done in odo node separate from this class
        skip_bin = None
        pose_error = None
        z_true = max(0., z_true) # insanity check
        if voc_ref.is_none():
            # choose voc_ref as depth with most valid measurements
            voc, voc_bin = self.find_voc_ref(z_true, t, roll, pitch, len(shear_list)-self.END_FILTER)
            # if no valid ref found, make new reference with 0 current and reset reference time
            if voc.is_none():
                print("new add_absolute_reference")
                voc_ref = OceanCurrent(0,0,0)
                # ignore returned pose error, as uncertainty is unknown without reference
                self.add_absolute_reference(z_true, t, voc_ref)
            # if voc is from shear bin, set voc_ref from shear value
            elif voc_bin is not None:
                skip_bin = voc_bin
                if voc_bin == -1:
                    voc_ref = voc
                else:
                    voc_ref = voc.subtract_shear(shear_list[voc_bin])
                    if not self.mag_filter(voc_ref, shear_list[voc_bin]):
                        voc_ref = OceanCurrent()
                    # print("voc: {}, voc_bin: {}, shear: {}, voc_ref: {}".format(voc, voc_bin, shear_list[voc_bin], voc_ref))
            # reference bin is current depth
            else:
                # print("voc_ref is top bin")
                voc_ref = voc
        else:
            # new absolute reference
            pose_error = self.add_absolute_reference(z_true, t, voc_ref)


        # iterate through shear list
        # Maybe START_FILTER should be 0 to propagate filtering... START_FILTER should be 1 when through water velocity estimated from bin0.
        # END_FILTER should be 1 or 2 to remove noisy bins near bottom, or bad deltas could be propagated
        if not voc_ref.is_none():
            for i in range(self.START_FILTER, len(shear_list)-self.END_FILTER):
                # find new bin for child node 
                child_z_true = self.get_z_true(z_true, roll, pitch, i)
                if child_z_true >= self.MAX_DEPTH:
                    continue
                child_z_bin  = self.get_wc_bin(child_z_true)
                child_voc    = voc_ref.add_shear(shear_list[i])

                # skip bin if specified
                if skip_bin!=i:
                    # only add child nodes with reasonable deltas 
                    if self.mag_filter(child_voc, shear_list[i]):
                        # shift old data over one column for new data. Update some column bins above and below child bin for smoothing
                        self.wc[child_z_bin,:,:] = np.roll(self.wc[child_z_bin,:,:], -1, 0)
                        # update water column bins at child depth +/- dvl depth bin size
                        voc_new = np.array([child_voc.u, child_voc.v, child_voc.w, t])
                        self.wc[child_z_bin,-1,:] = voc_new
            # Add voc estimate for reference water column bin from child shear
            if skip_bin is not None:
                # shift old data over one column for new data. Update some column bins above and below child bin for smoothing
                z_bin = self.get_wc_bin(z_true)
                self.wc[z_bin,:,:] = np.roll(self.wc[z_bin,:,:], -1, 0)
                # update water column bins at current depth +/- dvl depth bin size
                voc_new = np.array([voc_ref.u, voc_ref.v, voc_ref.w, t])
                self.wc[z_bin,-1,:] = voc_new

        return pose_error

    def add_absolute_reference(self, z_true, t, voc_ref):
        z_bin = self.get_wc_bin(z_true)
        voc, count = self.get_voc_at_depth(z_true, t)
        # back propagate if there are past measurements
        voc_err = OceanCurrent(0,0,0)
        if not voc.is_none():
            voc_err = voc_ref.subtract_shear(voc)

            # back propagate error to all water column measurements since last reference time
            wc_t = self.wc[:,:,3]
            wc_t = np.where(wc_t >= self.t_ref, 1, 0)
            count = wc_t.sum(1)
            row_indices = count.nonzero()[0]
            voc_new = np.array([voc_err.u, voc_err.v, voc_err.w])
            for i in range(len(row_indices)):
                col_indices = wc_t[i,:].nonzero()[0]
                voc_update = np.tile(voc_new, (len(col_indices),1))
                self.wc[row_indices[i],col_indices,:3] += voc_update

        # set full matrix rows at current depth +/- buffer to new absolute velocity measurement
        voc_new = np.array([voc_ref.u, voc_ref.v, voc_ref.w, t])
        voc_update = np.tile(voc_new, (self.wc.shape[1],1))
        self.wc[z_bin,:,:] = voc_update

        # update reference time and calculate pose error from elapsed time
        v_err = np.array([voc_err.u, voc_err.v, voc_err.w])
        # assume linear interpolation in velocity from 0 to voc_err since last fix.
        # this results in 1/2 the error estimate when assuming constant voc_error since
        # last fix
        if self.t_ref is not None:
            p_err = 0.5 * v_err * (t-self.t_ref)
        else:
            p_err = [0,0,0]
        self.t_ref = t

        return p_err

    def Qx(self, phi):
        """Orthogonal rotation matrix about x-axis by angle phi
        """
        return(np.array([[1,            0,            0],
                         [0,  np.cos(phi), -np.sin(phi)],
                         [0,  np.sin(phi),  np.cos(phi)]]))

    def Qy(self, phi):
        """Orthogonal rotation matrix about y-axis by angle phi
        """
        return(np.array([[ np.cos(phi), 0,  np.sin(phi)],
                         [           0, 1,            0],
                         [-np.sin(phi), 0,  np.cos(phi)]]))

    def Qz(self, phi):
        """Orthogonal rotation matrix about z-axis by angle phi
        """
        return(np.array([[ np.cos(phi), -np.sin(phi), 0],
                         [ np.sin(phi),  np.cos(phi), 0],
                         [           0,            0, 1]]))

### Unit Tests

def main():
    shear_length = 4
    bin_len = 2
    bin0_dist = 1.9
    max_depth = 30
    t = 0
    sample_number = 15

    np.set_printoptions(linewidth=np.inf)

    wc = WaterColumn(bin_len=bin_len,bin0_dist=bin0_dist,max_depth=max_depth,start_filter=0,end_filter=1,voc_mag_filter=10.0,voc_delta_mag_filter=0.20,voc_time_filter=10*60,sample_number=sample_number)

    def get_shear(z, b, pitch):
        v = OceanCurrent(.01*floor(z),-.01*floor(z),0)
        zb = floor(z + (bin0_dist+b*bin_len)*np.cos(pitch))
        vb = OceanCurrent(.01*zb,-.01*zb,0)
        return vb.subtract_shear(v)

    def gen_shear_list(z, pitch):
        shear_list = []
        # l = min(shear_length, floor((max_depth-z)/bin_len))
        l = shear_length
        for i in range(l):
            if (i*bin_len+bin0_dist)*np.cos(pitch) < bin_len:
                shear_list.append(OceanCurrent(0,0,0))
            else:
                shear_list.append(get_shear(z, i, pitch))
        return shear_list

    # virtual dive to 20 meters
    for z in np.arange(0,20,1):
    # for z in np.arange(0,20,0.2):
        # pitch = -30*np.pi/180.
        pitch = 0
        shear_list = gen_shear_list(z, pitch)
        wc.add_new_shear(z_true=z, t=t, shear_list=shear_list, voc_ref=OceanCurrent(), direction='descending', pitch=pitch, roll=0)
        t+=1

    # virtual ascend to 0 meters
    for z in np.arange(20,-0,-1):
        # pitch = 30*np.pi/180.
        pitch = 0
        shear_list = gen_shear_list(z, pitch)
        wc.add_new_shear(z_true=z, t=t, shear_list=shear_list, voc_ref=OceanCurrent(), direction='ascending', pitch=pitch, roll=0)
        t+=1

    print(wc.wc[:,:,0])
    print(wc.wc[:,:,1])
    print(wc.wc[:,:,3])

    wc.compute_averages()
    print(wc.averages_to_str())

if __name__ == "__main__":
    main()

