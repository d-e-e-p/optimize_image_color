#!/usr/bin/python3
#
#

import os, sys, re, math
from pathlib import Path
import numpy as np
import subprocess
import pdb


class Exp:
    def __init__(self, args):
        self.args = args
        if self.args.verbose:
            print("Init Exp ")
        # globals
        self.counter = 0
        self.min_x   = None
        self.min_res = None
        self.profile = None
        self.constraints = ()
        self.color_multiplier = 1.0
        self.grey_multiplier  = 1.0
        print(f"multipliers for (color,grey) = ({self.color_multiplier},{self.grey_multiplier})")

        # ok do the main setup in 3 steps: 
        SetupVars(self)
        SetupCallback(self)
        SetupGenProfiles(self)
        
    def whoami(self, from_mod ):
        print("hello {0}. I am instance {1}".format(from_mod, self))

class SetupVars:
    def __init__(self, exp):
        self.exp = exp
        #pdb.set_trace();
        # call the setup function based on operation variable
        func = getattr(self, self.exp.args.operation)
        func()

    def dng_wb(self):
        # best dng ccm after wb cf=11.5
        self.exp.x0 = np.array( [
                0.9, 
                0.9, 
        ])
        self.exp.bounds = [(0.4,2)] * self.exp.x0.size
        # only care about grey
        self.exp.color_multiplier = 0

    def dng_ccm(self):
        # best dng ccm after grey=20 deltaE 11.6
        x0 = np.array( [
                0.90273141, -0.23916363, -0.14392367, 
               -0.08329043,  1.1717975 , -0.11734164,  
                0.19868041,  0.25373908,  0.65681686,
        ])
        # grey=8 deltaE=14
        self.exp.x0 = np.array( [
             0.965, -0.368, -0.176, 
            -0.310,  0.977, -0.156, 
             0.117,  0.096,  1.430, 
        ])
        self.exp.bounds = [(-2,2)] * x0.size

    def rgb_equation_spline(self):

        # force inputs to be in increasing order
        #self.exp.process_inputs = getattr(self, 'force_monotonic')
        self.exp.constraints={"fun": constraint_monotonic, "type": "ineq"}

        x0 = np.array([
          0.161 , 0.409 , 0.678,
          0.256 , 0.513 , 0.800,
          0.259 , 0.502 , 0.728,
        ])
        x0 = np.array([
            0.151,  0.416,  0.737,
            0.250,  0.508,  0.768,
            0.256,  0.509,  0.740,
        ])
        x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
        ])
        self.exp.x0 = np.array([
         0.174 , 0.442  ,0.754 ,
         0.250 , 0.504  ,0.754 ,
         0.214 , 0.492  ,0.742 ,

        ])
        self.exp.bounds = [(0.1,0.9)] * self.exp.x0.size

    def rgb_equation_multi_ccm(self):
        self.exp.x0 = np.zeros( 16*3 )
        # cf = 7.5
        self.exp.x0 = np.array([
            7.07555110e-01, -4.18892216e-02,  5.21632256e-03, -5.46989948e-02,
            6.48380281e-01, -3.09466715e-02,  2.72643810e-02,  2.39125565e-02,
            6.95059686e-01, -3.11272118e-02,  9.72862211e-03, -3.74302259e-04,
            5.34734800e-01,  1.96765078e-01, -7.59708923e-02,  1.04621338e+00,
           -1.01966086e-01,  6.76919073e-01, -6.09138203e-02,  1.75397022e-02,
            2.16130374e-03,  6.61435494e-01,  5.15071269e-02,  1.16597968e-01,
            3.71452083e-02,  7.53564323e-01, -1.68554772e-03, -8.47147337e-03,
           -1.58792743e-01,  9.01003077e-01, -1.04256134e-01,  2.15004421e-02,
           -2.31891748e-02, -4.02681954e-02,  6.37127568e-01, -2.48251111e-02,
            4.46600298e-02,  4.19545904e-02,  7.10103578e-01, -1.96358505e-02,
            7.68817953e-02, -2.11231350e-02,  7.01171477e-01,  1.10196150e-01,
           -1.16677175e-01, -2.07322137e-01,  1.06732939e+00,  1.37196100e-02])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size


    def rgb_curves(self):

        # cf=11 but weird
        self.exp.x0 = np.array( [
          0.047 , 1.000,  1.000,
          0.237 , 0.000,  0.095,
          0.142 , 0.000,  0.192,
        ])
        self.exp.bounds = [(0,1)] * self.exp.x0.size

    def channel_mixer(self):

        # best Channel Mixer grey 8 deltaE 10
        self.exp.x0 = np.array( [
            -0.180,  0.070, -0.201, 
             0.011, -0.145, -0.179, 
            -0.165,  0.118, -0.252, 
        ])
        self.exp.bounds = [(-1,1)] * self.exp.x0.size


    def color_toning(self):

        # cf=18 or so..
        self.exp.x0 = np.array( [
            0.326,  0.755 , 1.000,
            0.158,  0.133 ,-0.016,
            0.194, -0.008 , 0.184,
        ])
        self.exp.bounds = [(-5,5)] * self.exp.x0.size


    def luminance_curve(self):
        self.exp.x0 = np.zeros( 12 )
        self.exp.x0 += 0.5
        self.exp.bounds = [(0.2,0.8)] * self.exp.x0.size


    def hsv_equalizer(self):
        
        self.exp.x0 = np.zeros( 30 )
        self.exp.x0 += 0.5
        self.exp.bounds = [(0,1)] * self.exp.x0.size

        # grey=7 and deltaE=11
        self.exp.x0 = np.array( [
         0.5,  0.5,  1.0,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,
         1.0,  1.0,  1.0,  0.5,  0.5,  0.5,  1.0,  0.5,  1.0,  1.0, 
         0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, 
        ])


class SetupCallback:
    def __init__(self, exp):
        self.exp = exp
        self.ops = Operation(exp)
        # just store function for later in Exp object

        self.exp.fun = getattr(self, self.exp.args.operation, None)
        if not self.exp.fun:
            self.exp.fun = getattr(self, 'generic_rt_with_profile')


    def dng_wb(self,x):
        # only allow R and B to vary holding G at 1
        args = f"""\"{x[0]} 1.0 {x[1]}\" """
        self.ops.run_cmd("/bin/rm -f images/corrected.dng ; exiftool -AsShotNeutral=" + args + " -o images/corrected.dng  images/img00006_G000E0400_wb_ccm.dng")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o images/corrected.jpg -c images/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def dng_ccm(self,x):
        args = get_args_string(x)

        self.ops.run_cmd("/bin/rm -f images/corrected.dng ; exiftool -ColorMatrix1=" + args + " -ColorMatrix2=" + args + " -o images/corrected.dng  images/img00006_G000E0400_wb.dng")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o images/corrected.jpg -c images/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def rgb_equation_spline(self,x):
        x = force_monotonic(x)
        args = get_args_string(x)
        self.ops.run_cmd("./bin/rawtherapee-cli -Y -o images/corrected.jpg -m " + args + " -c images/img00006_G000E0400_wb_ccm.dng")
        res = self.ops.check_results(x)
        return res

    def rgb_equation_multi_ccm(self,x):
        args = get_args_string(x)
        self.ops.run_cmd("./bin/rawtherapee-cli -Y -o images/corrected.jpg -m " + args + " -c images/img00006_G000E0400_wb_ccm.dng")
        res = self.ops.check_results(x)
        return res

    def generic_rt_with_profile(self,x):
        self.exp.gen_profile(x)
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -p profile.pp3 -o images/corrected.jpg -c images/img00006_G000E0400_wb_ccm.dng")
        res = self.ops.check_results(x)
        return res



class Operation:
    def __init__(self, exp):
        self.exp = exp

    def run_cmd(self, cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        #print(out)
        return out

    def check_results(self,x):
        out = self.run_cmd("./bin/macduff images/corrected.jpg images/check.jpg --restore")
        res = self.look_thru_cmd_output(x,out)
        self.exp.counter += 1
        if self.exp.args.tag is None:
            check_image = f"images_res/check_{self.exp.counter}.jpg"
        else:
            check_image = f"images_res/check_{self.exp.args.tag}_{self.exp.counter}.jpg"

        comment = f""" args:{self.exp.args},x:{x},res:{res} """
        cmd = f"""/bin/rm -f {check_image} ;  exiftool -o {check_image} -comment="{comment}" images/check.jpg"""
        self.run_cmd(cmd)
        return res

    def look_thru_cmd_output(self, x,out):
        # look for 'error = 85263.663537'
        m = re.search('grey_average_error *= *([0-9]*[.,]{0,1}[0-9]*)',str(out.stdout))
        if m:
            grey_average_error = float(m.group(1))
        else:
            grey_average_error = 5000.0

        m = re.search('deltaE_average_error *= *([0-9]*[.,]{0,1}[0-9]*)',str(out.stdout))
        if m:
            deltaE_average_error = float(m.group(1))
        else:
            deltaE_average_error = 5000.0

        # for white balance test ignore color metric .. focus on grey
        res = self.exp.grey_multiplier  * grey_average_error +  \
              self.exp.color_multiplier * deltaE_average_error

        if self.exp.min_res is None:
            self.exp.min_res = res
        self.exp.min_res = min(self.exp.min_res,res)
        if math.isclose(res, self.exp.min_res, rel_tol=1e-3):
            self.exp.min_x = x
        self.print_status(x,res)
        return res

    def print_status(self, x,res):

        class color:
           PURPLE = '\033[95m'
           CYAN = '\033[96m'
           DARKCYAN = '\033[36m'
           BLUE = '\033[94m'
           GREEN = '\033[92m'
           YELLOW = '\033[93m'
           RED = '\033[91m'
           BOLD = '\033[1m'
           UNDERLINE = '\033[4m'
           END = '\033[0m'

        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'
        size = int(x.size / 3)
            
        out = ""
        out += "res{}: current = {}{:.2f}{} min = {}{:.2f}{} ".format(self.exp.counter, color.BOLD, res, color.END, color.BOLD, self.exp.min_res, color.END)
        out += color.CYAN

        if (size > 0):
            m = k = 0
            for i in range(3):
              out += "\n"
              for j in range(size):
                out += "{: .3f}, ".format(x[k])
                k += 1
              out += "\t\t";
              for j in range(size):
                out += "{: .3f}, ".format(self.exp.min_x[m])
                m += 1
            out += "\n"
        else:
            for i in range(x.size):
                out += "\n"
                out += "{: .3f}, ".format(x[i])
                out += "\t\t";
                out += "{: .3f}, ".format(self.exp.min_x[i])
            out += "\n"
        out += color.END
                
        sys.stdout.write(out)
        sys.stdout.flush()
            


class SetupGenProfiles:
    def __init__(self, exp):
        self.exp = exp
        # store profile generator function in exp object, don't worry if no definition
        self.exp.gen_profile = getattr(self, self.exp.args.operation, None)

    def rgb_curves(self, x):
        # threshold to [0,1]
        x[x<0] = 0
        x[x>1] = 1

        write_rt_profile( f"""
# opt
[RGB Curves]
Enabled=true
LumaMode=true
rCurve=1;0;0;0.25;{x[0]};0.50;{x[1]};0.75;{x[2]};1;1;
gCurve=1;0;0;0.25;{x[3]};0.50;{x[4]};0.75;{x[5]};1;1;
bCurve=1;0;0;0.25;{x[6]};0.50;{x[7]};0.75;{x[8]};1;1;
    """)


    def color_toning(self, x):

        # precondition needed to improve root
        y = x * 1000

        write_rt_profile( f"""
# 
[ColorToning]
Enabled=true
Method=Splitco
Lumamode=true
Twocolor=Std
Autosat=true
Redlow=24
Greenlow=44
Bluelow=-31
Satlow=0
Balance=0
Sathigh=0
Redmed=-42
Greenmed=10
Bluemed=0
Redhigh=44
Greenhigh=0
Bluehigh=0
LabGridALow=0
LabGridBLow=0
LabGridAHigh=0
LabGridBHigh=0
LabRegionA_1=0
LabRegionB_1=0
LabRegionSaturation_1=0
LabRegionSlope_1=1
LabRegionOffset_1=0
LabRegionPower_1=1


[ColorToning]
Enabled=true
Method=Splitco
Redlow      = {int(y[0])}
Greenlow    = {int(y[1])}
Bluelow     = {int(y[2])}
Redmed      = {int(y[3])}
Greenmed    = {int(y[4])}
Bluemed     = {int(y[5])}
Redhigh     = {int(y[6])}
Greenhigh   = {int(y[7])}
Bluehigh    = {int(y[8])}
Satlow=0
Sathigh=0
Balance=0

""")

    def luminance_curve(x):

        write_rt_profile( f"""
# opt
[Luminance Curve]
Enabled=true
Brightness=  {int( y[9]*100 - 50)}
Contrast=    {int(y[10]*100 - 50)}
Chromaticity={int(y[11]*100 - 50)}
AvoidColorShift=false
RedAndSkinTonesProtection=0
LCredsk=true
LCurve =1;0;0;0.5;{y[0]};1;1;
aCurve =1;0;0;0.5;{y[1]};1;1;
bCurve =1;0;0;0.5;{y[2]};1;1;
ccCurve=1;0;0;0.5;{y[3]};1;1;
chCurve=1;0;0;0.5;{y[4]};1;1;
lhCurve=1;0;0;0.5;{y[5]};1;1;
hhCurve=1;0;0;0.5;{y[6]};1;1;
LcCurve=1;0;0;0.5;{y[7]};1;1;
ClCurve=1;0;0;0.5;{y[8]};1;1;
""")


    def channel_mixer(x):

        # needed for COBYLA
        y = x * 1000

        write_rt_profile( f"""
# opt
[Channel Mixer]
Enabled=true
Red={int(1000+y[0])};{int(y[1])};{int(y[2])};
Green={int(y[3])};{int(1000+y[4])};{int(y[5])}
Blue={int(y[6])};{int(y[7])};{int(1000+y[8])}

""")

    def hsv_equalizer(x):

        # threshold to [0,1]
        y = x
        y[y<0] = 0
        y[y>1] = 1

        HCurve = f"""
       0.0;{y[0]};0.2;0.5;
       0.1;{y[1]};0.2;0.5;
       0.2;{y[2]};0.2;0.5;
       0.3;{y[3]};0.2;0.5;
       0.4;{y[4]};0.2;0.5;
       0.5;{y[5]};0.2;0.5;
       0.6;{y[6]};0.2;0.5;
       0.7;{y[7]};0.2;0.5;
       0.8;{y[8]};0.2;0.5;
       0.9;{y[9]};0.2;0.5;
    """.replace('\n', '')

        SCurve = f"""
       0.0;{y[10]};0.2;0.5;
       0.1;{y[11]};0.2;0.5;
       0.2;{y[12]};0.2;0.5;
       0.3;{y[13]};0.2;0.5;
       0.4;{y[14]};0.2;0.5;
       0.5;{y[15]};0.2;0.5;
       0.6;{y[16]};0.2;0.5;
       0.7;{y[17]};0.2;0.5;
       0.8;{y[18]};0.2;0.5;
       0.9;{y[19]};0.2;0.5;
    """.replace('\n', '')

        VCurve = f"""
       0.0;{y[20]};0.2;0.5;
       0.1;{y[21]};0.2;0.5;
       0.2;{y[22]};0.2;0.5;
       0.3;{y[23]};0.2;0.5;
       0.4;{y[24]};0.2;0.5;
       0.5;{y[25]};0.2;0.5;
       0.6;{y[26]};0.2;0.5;
       0.7;{y[27]};0.2;0.5;
       0.8;{y[28]};0.2;0.5;
       0.9;{y[29]};0.2;0.5;
    """.replace('\n', '')

        write_rt_profile( f"""
# opt
[HSV Equalizer]
Enabled=true
HCurve=1; {HCurve}
SCurve=1; {SCurve}
VCurve=1; {VCurve}
""")


#
# common functions
#


def get_args_string(x):
    # preprocess x 
    #if profile is None:
    #    y = x
    #else:
    #   y = profile(x)
    args = "\"";
    for i in range(x.size):
        args += str(x[i]) + " ";
    args += "\"";
    return args

def force_monotonic(x):
    # limit to range 0,1
    x[x<0] = 0
    x[x>1] = 1
    # force increasing
    y = np.array(x)
    for i in range(1, y.size):
        if y[i] < y[i-1]:
            y[i] = y[i-1]
    return y

# inequality means return has to be positive to be accepted
def constraint_monotonic(x):
    if (x < 0).any():
        return -1
    if (x > 1).any():
        return -1
    # force increasing
    for i in range(1, x.size):
        if x[i] < x[i-1]:
            return -1
    return 0


def write_rt_profile(out):
    fp = open('profile.pp3', 'w')
    fp.write(out)
    fp.close()


