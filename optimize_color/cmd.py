#!/usr/bin/python3
#
#

import os, sys, re, math
from pathlib import Path
import numpy as np
import subprocess
import pdb
import shutil

import rich
from rich.console import Console
from rich.table import Table


class Exp:
    def __init__(self, args, operation):
        self.args = args
        self.operation = operation
        terminal = shutil.get_terminal_size((132, 40))
        self.console = Console(width=terminal.columns, force_terminal=True)

        # ok do the main init setup in 4 easy steps: 
        self.process_args()     # process args -> Exp object
        SetupVars(self)         # variables for each operation
        SetupCallback(self)     # define cost function
        SetupGenProfiles(self)  # profiles for running some functions
        if self.args.verbose:
            print(f"Init Exp with operation: {operation} ")

    def process_args(self):
        # globals
        self.counter = 0
        self.min_x   = None
        self.min_res = None
        self.profile = None
        self.blacklevelp = None
        self.whitelevelp = None
        if self.args.tag is None:
            self.args.tag = self.operation
        if self.args.src is None:
            self.src = "images/img00006_G000E0400_wb_ccm.dng"
        else:
            self.src = self.args.src
        if self.args.dst is None:
            self.dst = None
        else:
            self.dst = self.args.dst
        self.constraints = ()   # can't be None

        self.color_multiplier = 1.0
        self.grey_multiplier  = 0.0
        print(f"default multipliers for (color,grey) = ({self.color_multiplier},{self.grey_multiplier})")
        
    def whoami(self, from_mod ):
        print("hello {0}. I am instance {1}".format(from_mod, self))

# end class Exp

class SetupVars:
    def __init__(self, exp):
        self.exp = exp
        # call the setup function based on operation variable, eg dng_wb..must exist
        func = getattr(self, self.exp.operation)
        func()

    def dng_wb(self):
        # best dng ccm after wb cf=11.5
        self.exp.x0 = np.array( [
                0.8, 
                0.6, 
        ])
        self.exp.bounds = [(0.4,2)] * self.exp.x0.size
        # only really care about grey?
        self.exp.grey_multiplier  = 1.0
        self.exp.color_multiplier = 0.5

    def dng_ccm(self):
        # best dng ccm after grey=20 deltaE 11.6
        x0 = np.array( [
                0.90273141, -0.23916363, -0.14392367, 
               -0.08329043,  1.1717975 , -0.11734164,  
                0.19868041,  0.25373908,  0.65681686, 0.2, 2 
        ])
        # grey=8 deltaE=14
        self.exp.x0 = np.array( [
             2.521,-0.456,-0.199,
            -0.416, 1.210, 0.174,
            -0.212, 0.053, 3.328, 2.141, 0.008 
        ])
        # for example2
        self.exp.x0 = np.array([
            1.,  0,  0,
            0.,  1,  0,
            0.,  0,  1,  1, 1
        ])
        self.exp.x0 = np.array([
            0.970, 0.003, 0.003, 
            0.003, 0.999,-0.001, 
            0.006, 0.006, 1.006, 0.975, 1.150
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
        self.exp.x0 = np.array([
         0.174 , 0.442  ,0.754 ,
         0.250 , 0.504  ,0.754 ,
         0.214 , 0.492  ,0.742 ,

        ])
        self.exp.x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
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
        self.exp.constraints={"fun": constraint_monotonic, "type": "ineq"}
        self.exp.x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
        ])
        self.exp.x0 = np.array( [
             0.268, 0.409, 0.531, 
             0.273, 0.488, 0.832, 
             0.282, 0.490, 0.752,
        ])
        self.exp.bounds = [(0.1,0.9)] * self.exp.x0.size

    def channel_mixer(self):
        # best Channel Mixer grey 8 deltaE 10
        self.exp.x0 = np.array( [
	-0.258,  0.114, -0.149,
	 0.024, -0.207, -0.170,
	-0.183,  0.174, -0.342,
        ])
        self.exp.x0 = np.zeros( 9 )
        self.exp.bounds = [(-1,1)] * self.exp.x0.size


    def color_toning(self):
        # cf=18 or so..
        self.exp.x0 = np.array( [
            0.326,  0.755 , 1.000,
            0.158,  0.133 ,-0.016,
            0.194, -0.008 , 0.184,
        ])
        self.exp.x0 = np.zeros( 9 )
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
        #self.exp.x0 = np.array( [
        # 0.5,  0.5,  1.0,  0.5,  0.5,  1.0,  0.5,  0.5,  0.5,  0.5,
        # 1.0,  1.0,  1.0,  0.5,  0.5,  0.5,  1.0,  0.5,  1.0,  1.0, 
        # 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5, 
        #])
# end class SetupVars


class SetupCallback:
    def __init__(self, exp):
        self.exp = exp
        self.ops = Operation(exp)
        # just store function for later in Exp object

        self.exp.fun = getattr(self, self.exp.operation, None)
        if not self.exp.fun:
            self.exp.fun = getattr(self, 'generic_rt_with_profile')


    def dng_wb(self,x):
        # only allow R and B to vary holding G at 1
        args = f"""\"{x[0]} 1.0 {x[1]}\" """
        
        self.ops.run_cmd(f"/bin/rm -f images/corrected.dng ; exiftool -AsShotNeutral={args} -o images/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o images/corrected.jpg -c images/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def dng_ccm(self,x):
        # extra conf at end for blacklevel
        y = x[0:9:1] # first 9 elements only
        args = get_args_string(y)
        # hack
        #x[9] = 1
        #x[10] = 1

        # assume 16bit images
        # for eg2 Black=1024 White=15600
        sixteenbit = 2**16 - 1
        #blacklevel = int(x[9]  * (4000.0/2.0)) # blacklevel limit should be around 4000
        #whitelevel = sixteenbit - int(x[10] * (40000.0/2.0)) # whitelevel 
        blacklevel = int(x[9]  * 1024) # blacklevel = 1024
        whitelevel = sixteenbit - int(x[10] * 49935) # whitelevel = 15600

        self.exp.blacklevelp =  blacklevel * 100.0/sixteenbit
        self.exp.whitelevelp =  whitelevel * 100.0/sixteenbit

        #self.ops.run_cmd("/bin/rm -f images/corrected.dng ; exiftool -ColorMatrix1=" + args + " -ColorMatrix2=" + args + " -o images/corrected.dng  images/img00006_G000E0400_wb.dng")
        self.ops.run_cmd(f"/bin/rm -f images/corrected.dng ; exiftool -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args} -ColorMatrix2={args} -IFD0:BlackLevel={blacklevel} -IFD0:WhiteLevel={whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={blacklevel} -SubIFD:WhiteLevel={whitelevel}  -o images/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o images/corrected.jpg -c images/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def rgb_equation_spline(self,x):
        x = force_monotonic(x)
        return self.generic_rt_with_x_arguments(x)

    def rgb_equation_multi_ccm(self,x):
        return self.generic_rt_with_x_arguments(x)

    def rgb_curves(self,x):
        x = force_monotonic(x)
        return self.generic_rt_with_profile(x)

    def generic_rt_with_x_arguments(self,x):
        args = get_args_string(x)
        self.ops.run_cmd(f"./bin/rawtherapee-cli -Y -o images/corrected.jpg -m {args} -c {self.exp.src}")
        return self.ops.check_results(x)

    def generic_rt_with_profile(self,x):
        self.exp.gen_profile(x)
        self.ops.run_cmd(f"/usr/local/bin/rawtherapee-cli -Y -p profile.pp3 -o images/corrected.jpg -c {self.exp.src}")
        return self.ops.check_results(x)

# end class SetupCallback


class Operation:
    def __init__(self, exp):
        self.exp = exp

    def run_cmd(self, cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        if self.exp.args.verbose:
            print(out)
        return out

    def check_results(self,x):
        #out = self.run_cmd("./bin/macduff images/corrected.jpg images/check.jpg --restore")
        out = self.run_cmd("./bin/macduff_example2 images/corrected.jpg images/check.jpg --restore")
        res = self.look_thru_cmd_output(x,out)
        # store check image with attributes
        self.exp.counter += 1
        check_image = f"images_res/check_{self.exp.args.tag}_{self.exp.counter}.jpg"
        comment = f""" args:{self.exp.args},x:{x},grey_average_error:{self.exp.grey_average_error},deltaE_average_error{self.exp.deltaE_average_error},res:{res} """
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

        self.exp.grey_average_error = grey_average_error
        self.exp.deltaE_average_error = deltaE_average_error

        if self.exp.min_res is None:
            self.exp.min_res = res
        self.exp.min_res = min(self.exp.min_res,res)

        # record this state if res is optimal
        if math.isclose(res, self.exp.min_res, rel_tol=1e-3):
            self.exp.min_x = x
            self.exp.min_grey_average_error   = grey_average_error
            self.exp.min_deltaE_average_error = deltaE_average_error
            if self.exp.blacklevelp is not None:
                self.exp.min_blacklevelp = self.exp.blacklevelp
                self.exp.min_whitelevelp = self.exp.whitelevelp

        self.exp.res = res
        self.print_status(x)
        return res

    def print_status(self, cur_x):

        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'

        # first define all the text
        text_title = f"op={self.exp.operation} method={self.exp.method}"
        text_run = f"run{self.exp.counter}"

        text_cur_error =  f"(grey,color) {self.exp.grey_multiplier} X {self.exp.grey_average_error:4.1f} + {self.exp.color_multiplier} X {self.exp.deltaE_average_error:4.1f} = {self.exp.res:5.2f}"
        text_min_error =  f"(grey,color) {self.exp.grey_multiplier} X {self.exp.min_grey_average_error:4.1f} + {self.exp.color_multiplier} X {self.exp.min_deltaE_average_error:4.1f} = {self.exp.min_res:5.2f}" 


        if self.exp.blacklevelp is not None:
            text_cur_level = f"(blacklevel,whitelevel) = ({self.exp.blacklevelp:.2f}%,{self.exp.whitelevelp:.2f}%)"
            text_min_level = f"(blacklevel,whitelevel) = ({self.exp.min_blacklevelp:.2f}%,{self.exp.min_whitelevelp:.2f}%)"

        text_cur_x = np.array2string(cur_x,    
            max_line_width=1000,precision=3,separator=",",floatmode='fixed',sign=' ', formatter={'float_kind':lambda x: "% 4.3f" % x})
        text_min_x = np.array2string(self.exp.min_x,
            max_line_width=1000,precision=3,separator=",",floatmode='fixed',sign=' ', formatter={'float_kind':lambda x: "% 4.3f" % x})

        # needed for bug in rich table
        text_cur_x = text_cur_x.strip(']').strip('[')
        text_min_x = text_min_x.strip(']').strip('[')

        # ok we're now ready to define the table
        table = Table(title=text_title, show_lines=True, show_header=False, )
        #table.add_column("current", style="cyan", justify="center")
        #table.add_column("best", style="yellow", justify="center")

        if self.exp.blacklevelp is not None:
            table.add_row(text_run, text_cur_error, text_cur_level, text_cur_x)
            table.add_row("best"   , text_min_error, text_min_level, text_min_x)
        else:
            table.add_row(text_run, text_cur_error, text_cur_x)
            table.add_row("best"   , text_min_error, text_min_x)

            
        self.exp.console.print(table)


# end class Operation
            


class SetupGenProfiles:
    def __init__(self, exp):
        self.exp = exp
        # store profile generator function in exp object, don't worry if no definition
        self.exp.gen_profile = getattr(self, self.exp.operation, None)

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

    def luminance_curve(self,x):

        write_rt_profile( f"""
# opt
[Luminance Curve]
Enabled=true
Brightness=  {int( x[9]*100 - 50)}
Contrast=    {int(x[10]*100 - 50)}
Chromaticity={int(x[11]*100 - 50)}
AvoidColorShift=false
RedAndSkinTonesProtection=0
LCredsk=true
LCurve =1;0;0;0.5;{x[0]};1;1;
aCurve =1;0;0;0.5;{x[1]};1;1;
bCurve =1;0;0;0.5;{x[2]};1;1;
#ccCurve=1;0;0;0.5;{x[3]};1;1;
#chCurve=1;0;0;0.5;{x[4]};1;1;
#lhCurve=1;0;0;0.5;{x[5]};1;1;
#hhCurve=1;0;0;0.5;{x[6]};1;1;
#LcCurve=1;0;0;0.5;{x[7]};1;1;
#ClCurve=1;0;0;0.5;{x[8]};1;1;
""")


    def channel_mixer(self,x):

        # needed for COBYLA
        y = x * 1000

        write_rt_profile( f"""
# opt
[Channel Mixer]
Enabled=true
Red={int(1000+x[0])};{int(y[1])};{int(y[2])};
Green={int(x[3])};{int(1000+y[4])};{int(y[5])}
Blue={int(x[6])};{int(y[7])};{int(1000+y[8])}

""")

    def hsv_equalizer(self,x):

        # threshold to [0,1]
        x[x<0] = 0
        x[x>1] = 1

        HCurve = f"""
       0.0;{x[0]};0.35;0.35;
       0.1;{x[1]};0.35;0.35;
       0.2;{x[2]};0.35;0.35;
       0.3;{x[3]};0.35;0.35;
       0.4;{x[4]};0.35;0.35;
       0.5;{x[5]};0.35;0.35;
       0.6;{x[6]};0.35;0.35;
       0.7;{x[7]};0.35;0.35;
       0.8;{x[8]};0.35;0.35;
       0.9;{x[9]};0.35;0.35;
    """.replace('\n', '').strip()

        SCurve = f"""
       0.0;{x[10]};0.35;0.35;
       0.1;{x[11]};0.35;0.35;
       0.2;{x[12]};0.35;0.35;
       0.3;{x[13]};0.35;0.35;
       0.4;{x[14]};0.35;0.35;
       0.5;{x[15]};0.35;0.35;
       0.6;{x[16]};0.35;0.35;
       0.7;{x[17]};0.35;0.35;
       0.8;{x[18]};0.35;0.35;
       0.9;{x[19]};0.35;0.35;
    """.replace('\n', '').strip()

        VCurve = f"""
       0.0;{x[20]};0.35;0.35;
       0.1;{x[21]};0.35;0.35;
       0.2;{x[22]};0.35;0.35;
       0.3;{x[23]};0.35;0.35;
       0.4;{x[24]};0.35;0.35;
       0.5;{x[25]};0.35;0.35;
       0.6;{x[26]};0.35;0.35;
       0.7;{x[27]};0.35;0.35;
       0.8;{x[28]};0.35;0.35;
       0.9;{x[29]};0.35;0.35;
    """.replace('\n', '').strip()

        write_rt_profile( f"""
# opt
[HSV Equalizer]
Enabled=true
HCurve=1;{HCurve}
SCurve=1;{SCurve}
VCurve=1;{VCurve}
""")

# end class SetupGenProfiles

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

def force_array_monotonic(x):
    for i in range(1, x.size):
        if x[i] < x[i-1]:
            x[i] = x[i-1]
    return x


def force_monotonic(x):
    # limit to range 0,1
    x[x<0] = 0
    x[x>1] = 1

    # force increasing for every 3 elements
    yr = x[0:3:1]
    yg = x[3:6:1]
    yb = x[6:9:1]

    yr = force_array_monotonic(yr)
    yg = force_array_monotonic(yg)
    yb = force_array_monotonic(yb)

    y = np.concatenate([yr,yg,yb])
    return y

def array_not_monotonic(x):
    for i in range(1, x.size):
        if x[i] < x[i-1]:
            return True
    return False

# inequality means return has to be positive to be accepted
def constraint_monotonic(x):
    if (x < 0).any():
        return -1
    if (x > 1).any():
        return -1
    # force increasing
    yr = x[0:3:1]
    yg = x[3:6:1]
    yb = x[6:9:1]
    if array_not_monotonic(yr):
        return -1
    if array_not_monotonic(yg):
        return -1
    if array_not_monotonic(yb):
        return -1
    #pdb.set_trace();
    return 1


def write_rt_profile(out):
    fp = open('profile.pp3', 'w')
    fp.write(out)
    fp.close()


