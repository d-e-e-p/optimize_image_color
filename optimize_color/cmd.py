#!/usr/bin/python3
#
#

import os, sys, re, math
import os.path
import numpy as np
import subprocess
import pudb
import shutil
from collections import defaultdict

import rich
from rich.console import Console
from rich.table import Table

from json_tricks import dumps, loads


class Exp:
    def __init__(self, args, operation):
        self.args = args
        self.operation = operation
        terminal = shutil.get_terminal_size((1024, 40))
        self.console = Console(width=terminal.columns, force_terminal=True)

        # setup output dirs
        os.makedirs("run/check", exist_ok = True) 

        # ok do the main init setup in 4 easy steps: 
        self.process_args()     # process args -> Exp object
        SetupVars(self)         # variables for each operation
        SetupCallback(self)     # define cost function
        SetupGenProfiles(self)  # profiles for running some functions
        self.restore_best()
        
        if self.args.verbose:
            print(f"Init Exp with tag: {self.tag} ")

    def process_args(self):
        # globals
        self.counter = 0
        self.min_x   = None
        self.min_res = None
        self.min_blacklevelp = None
        self.min_whitelevelp = None
        self.check_image = None
        self.profile = None
        self.blacklevelp = None
        self.whitelevelp = None

        if self.args.src is None:
            self.src = "inputs/images/img00006_G000E0400.dng"
        else:
            self.src = self.args.src

        if self.args.dst is None:
            self.dst = get_destination_from_operation(self.src, self.operation)
        else:
            self.dst = self.args.dst

        self.csv = get_csv_from_src(self.src)
        print(f"src: {self.src} -> dst: {self.dst}")

        # mark this run
        if self.args.tag is None:
            basename = os.path.basename(self.src).rsplit('.',1)[0]
            self.tag = f"{basename}_{self.operation}"
        else:
            self.tag = self.args.tag

        self.constraints = ()   # can't be None

        self.cost_multiplier = {'greyFC_average': 0, 'deltaE_average': 1, 'deltaC_average': 0} 
        print(f"default multipliers : {self.cost_multiplier}")
        
    #
    # save best results
    #
    def save_best(self):

        # save the error image
        jsonfile, imgfile, exiffile = self.get_save_restore_filename(self.args.src, self.operation)
        copy2(self.check_image, imgfile)

        srcdng = "run/corrected.dng"
        if self.dst is not None:
            copy2(srcdng,self.dst)
            if exiffile is not None:
                # pu.db
                self.ops.run_cmd(f"exiftool -args {self.dst} > {exiffile}")

        # save the corrected image

        data = self.get_image_data_for_json()
        json = dumps(data, indent=4)

        fp = open(jsonfile, 'w')
        fp.write(json)
        fp.close()

    def get_image_data_for_json(self):
        save_fields = """
            args
            operation
            method
            src
            dst
            tag
            csv
            
            counter
            cost_multiplier
            bounds
            profile
            
            min_x
            min_blacklevelp
            min_whitelevelp
            min_error
            min_res
        """.split()

        data = {}
        for s in save_fields:
            data[s] = getattr(self, s, None) 
        return data

    #
    # restore results if exists
    #
    def restore_best(self):

        jsonfile, imgfile, dngfile = self.get_save_restore_filename(self.args.src, self.operation)
        if not os.path.isfile(jsonfile): 
            return

        fp = open(jsonfile, 'r')
        json = fp.read()
        fp.close()

        data = loads(json)
        self.x0 = data['min_x']
        print(f"restored results from file {jsonfile} :")
        print(f"x0 = {self.x0}")

    def get_save_restore_filename(self, src, operation):
        base = src.rsplit('.',1)[0]
        basename = os.path.basename(base)
        dirname  = f"results/{basename}"
        jsonfile = f"{dirname}/{operation}.json"
        imgfile  = f"{dirname}/{operation}.jpg"
        if operation.startswith("dng_"):
            exiffile  = f"{dirname}/{operation}.exif"
        else:
            exiffile  = None
        os.makedirs(dirname, exist_ok = True) 
        return jsonfile, imgfile, exiffile

    def whoami(self, from_mod ):
        print("hello {0}. I am instance {1}".format(from_mod, self))

# end class Exp

class SetupVars:
    def __init__(self, exp):
        self.exp = exp
        # call the setup function based on operation variable, eg dng_wb..must exist
        func = getattr(self, self.exp.operation)
        if not func:
            print(f"ERROR: SetupVars not found for operation {self.exp.operation}")
            breakpoint();
        func()

    def dng_wb(self):
        self.exp.x0 = np.array( [
                1.0, 
                1.0, 
        ])
        self.exp.bounds = [(0.2,2)] * self.exp.x0.size
        self.exp.constraints={"fun": constraint_positive, "type": "ineq"}
        self.exp.cost_multiplier = {'greyFC_average': 0.9, 'deltaE_average': 0, 'deltaC_average': 0.1} 
        print(f"reset multipliers to : {self.exp.cost_multiplier}")

    def dng_wb_bl(self):
        self.exp.x0 = np.array( [
                1, 1,       # wb 
                0, 0,       # bl, wl
        ])
        self.exp.bounds = [(0,2)] * self.exp.x0.size
        self.exp.constraints={"fun": constraint_positive, "type": "ineq"}
        self.exp.cost_multiplier = {'greyFC_average': 0.9, 'deltaE_average': 0, 'deltaC_average': 0.1} 
        print(f"reset multipliers to : {self.exp.cost_multiplier}")

    def dng_ccm(self):
        # starting point
        self.exp.x0 = np.array([
            1.,  0,  0,
            0.,  1,  0,
            0.,  0,  1,
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def dng_ccm_bl(self):
        self.exp.x0 = np.array([
            1.,  0,  0,
            0.,  1,  0,
            0.,  0,  1,  0, 0
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def dng_wb_ccm(self):
        self.exp.x0 = np.array([
            1, 1,          #  wb 
            1,  0,  0,     #  ccm
            0,  1,  0,
            0,  0,  1,
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size
        self.exp.cost_multiplier = {'greyFC_average': 0.1, 'deltaE_average': 0.9, 'deltaC_average': 0.0} 
        print(f"reset multipliers to : {self.exp.cost_multiplier}")

    def dng_wb_ccm_bl(self):
        self.exp.x0 = np.array([
            1, 1,          #  wb 
            1,  0,  0,     #  ccm
            0,  1,  0,     #  ccm 
            0,  0,  1,     #  ccm 
            0, 0,          #  bl,wl
        ])
        self.exp.x0 = np.array([
            0.8, 0.4,          #  wb 
            1,  0,  0,     #  ccm
            0,  1,  0,     #  ccm 
            0,  0,  1,     #  ccm 
            0.8, 0.1,          #  bl,wl
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size
        self.exp.cost_multiplier = {'greyFC_average': 0.1, 'deltaE_average': 0.9, 'deltaC_average': 0.0} 
        print(f"reset multipliers to : {self.exp.cost_multiplier}")

    def eq_spline(self):

        # force inputs to be in increasing order in calling function
        self.exp.constraints={"fun": constraint_monotonic, "type": "ineq"}

        self.exp.x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
        ])

        if "CanonG10ColorTestChart" in self.exp.src:
            self.exp.x0 = np.array([
                0.117, 0.481, 0.726, 0.332, 0.526, 0.747, 0.242, 0.523, 0.743 
            ])

        if "Df_WB_Demo006" in self.exp.src:
            self.exp.x0 = np.array([
                0.309, 0.612, 1.000, 0.263, 0.695, 1.000, 0.369, 0.567, 1.000 
            ])

        if "img00006_" in self.exp.src:
            self.exp.x0 = np.array([
                0.250, 0.505, 0.741, 
                0.247, 0.508, 0.750, 
                0.247, 0.503, 0.767,
            ])


        self.exp.bounds = [(0.1,0.9)] * self.exp.x0.size

    def eq_root_polynomial(self):
        self.exp.x0 = np.array([
            1.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
            0.,  1,  0,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
            0.,  0,  1,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
        ])
        if "img00006_" in self.exp.src:
            self.exp.x0 = np.array([
 1.000,-0.004,-0.004,-0.010,-0.010,-0.012,-0.005, 0.003, 0.002,-0.003, 0.006, 0.003, 0.003,
-0.009, 1.006,-0.010,-0.002,-0.002,-0.008,-0.010,-0.001,-0.007,-0.007,-0.004, 0.002, 0.005,
-0.011,-0.008, 0.996, 0.001, 0.000,-0.005,-0.010,-0.002, 0.095,-0.076,-0.006,-0.003,-0.006,
            ])

        # cf = 7.5
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def eq_multi_ccm2(self):
        self.exp.x0 = np.array([
            1, 0, 0,  1, 0, 0, 
            0, 1, 0,  0, 1, 0, 
            0, 0, 1,  0, 0, 1, 
        ])
        if "img00006_" in self.exp.src:
            self.exp.x0 = np.array([
             0.975, 0.001,-0.032, 1.009,-0.000, 0.003,
             0.003, 0.942,-0.038, 0.006, 1.009, 0.008,
            -0.029,-0.029, 1.002, 0.006, 0.003, 1.005,
        ])

        #if "img00006_" in self.exp.src:

        # cf = 7.5
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def eq_multi_ccm4(self):
        self.exp.x0 = np.array([
            1,0,0, 1,0,0, 1,0,0, 1,0,0,
            0,1,0, 0,1,0, 0,1,0, 0,1,0,
            0,0,1, 0,0,1, 0,0,1, 0,0,1,
        ])
        # cf = 7.5
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def eq_gamma_scale(self):
        self.exp.x0 = np.array([
            1.,  0,  0,  0.,  0,  0,  0,
            0.,  1,  0,  0.,  0,  0,  0,
            0.,  0,  1,  0.,  0,  0,  0,
        ])
        # cf = 7.5
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def rt_rgb_curves(self):
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

    def rt_channel_mixer(self):
        # best Channel Mixer grey 8 deltaE 10
        self.exp.x0 = np.array( [
        -0.258,  0.114, -0.149,
         0.024, -0.207, -0.170,
        -0.183,  0.174, -0.342,
        ])
        self.exp.x0 = np.zeros( 9 )
        self.exp.bounds = [(-1,1)] * self.exp.x0.size


    def rt_color_toning(self):
        # cf=18 or so..
        self.exp.x0 = np.array( [
            0.326,  0.755 , 1.000,
            0.158,  0.133 ,-0.016,
            0.194, -0.008 , 0.184,
        ])
        self.exp.x0 = np.zeros( 9 )
        self.exp.bounds = [(-5,5)] * self.exp.x0.size


    def rt_luminance_curve(self):
        self.exp.x0 = np.zeros( 12 )
        self.exp.x0 += 0.5
        self.exp.bounds = [(0.2,0.8)] * self.exp.x0.size


    def rt_hsv_equalizer(self):
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
        self.exp.ops = self.ops # alias
        # just store function for later in Exp object
        self.exp.fun = getattr(self, self.exp.operation, None)
        if not self.exp.fun:
            if self.exp.operation.find("rt_") == 0 :
                self.exp.fun = getattr(self, 'rt_generic')
            elif self.exp.operation.find("eq_") == 0 :
                self.exp.fun = getattr(self, 'eq_generic')
            else:
                print(f"ERROR: handler not found for operation {self.exp.operation}")
                breakpoint();

    def set_blackwhite_level(self, bl, wl):
        # assume 16bit images
        # TODO: make it work for 8-bit inputs
        # for eg2 Black=1024 White=15600
        bl = clamp(bl,0,2)
        wl = clamp(wl,0,2)
        sixteenbit = 2**16 - 1

        # bl=2 makes black=10% while wl=2 makes white=20%
        self.exp.blacklevelp = 5 * bl 
        self.exp.whitelevelp = 100 - (40 * wl)
        self.exp.blacklevel  = int(sixteenbit * self.exp.blacklevelp / 100.0)
        self.exp.whitelevel  = int(sixteenbit * self.exp.whitelevelp / 100.0)


    def dng_wb(self,x):

        # only allow R and B to vary holding G at 1
        r,g,b = (x[0], 1, x[1])

        #force r and b to be positive
        if r < 0: return 100 - 100 * r 
        if b < 0: return 100 - 100 * b 
            
        args = f"""\"{r} {g} {b}\" """
        
        self.ops.run_cmd(f"/bin/rm -f run/corrected.dng ; exiftool -AsShotNeutral={args} -o run/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.ops.check_results(x)
        return res

    # white balance with blacklevel/whitelevel
    def dng_wb_bl(self,x):

        # only allow R and B to vary holding G at 1
        r,g,b = (x[0], 1, x[1])
        self.set_blackwhite_level(x[2], x[3])

        #force r and b to be positive
        if r < 0:
            res = 100 - 100 * r 
            return res

        if b < 0:
            res = 100 - 100 * b 
            return res
            
        args = f"""\"{r} {g} {b}\" """
        self.ops.run_cmd(f"/bin/rm -f run/corrected.dng ; exiftool -AsShotNeutral={args} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def dng_ccm (self,x):
        args = get_args_string(x)
        self.ops.run_cmd(f"/bin/rm -f run/corrected.dng ; exiftool -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args} -ColorMatrix2={args} -o run/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def dng_ccm_bl(self,x):
        # extra conf at end for blacklevel
        y = x[0:9:1] # first 9 elements only
        args = get_args_string(y)
        self.set_blackwhite_level(x[9], x[10])

        self.ops.run_cmd(f"/bin/rm -f run/corrected.dng ; exiftool -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args} -ColorMatrix2={args} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def dng_wb_ccm_bl(self,x):

        # first 2 elements are for wb
        r,g,b = (x[0], 1, x[1])
        args_wb = f"""\"{r} {g} {b}\" """

        #force r and b to be positive
        if r < 0: return 100 - 100 * r 
        if b < 0: return 100 - 100 * b 

        # next 9 elements for ccm
        y = x[2:11:1] 
        args_ccm = get_args_string(y)

        # last 2 elements for bl,wl
        self.set_blackwhite_level(x[11], x[12])

        self.ops.run_cmd(f"/bin/rm -f run/corrected.dng ; exiftool -AsShotNeutral={args_wb}  -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args_ccm} -ColorMatrix2={args_ccm} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.src}")
        self.ops.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.ops.check_results(x)
        return res

    def eq_spline(self,x):
        x = force_monotonic(x)
        return self.eq_generic(x)

    def rt_generic(self,x):
        self.exp.gen_profile(x)
        self.ops.run_cmd(f"/usr/local/bin/rawtherapee-cli -Y -p profile.pp3 -o run/corrected.jpg -c {self.exp.src}")
        return self.ops.check_results(x)

    def eq_generic(self,x):
        args = get_args_string(x)
        self.ops.run_cmd(f"./bin/rawtherapee-cli -Y -o run/corrected.jpg -e {self.exp.operation} -m {args} -c {self.exp.src}")
        return self.ops.check_results(x)

# end class SetupCallback


class Operation:
    def __init__(self, exp):
        self.exp = exp
        self.text = {}

    def run_cmd(self, cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # TODO: stop rt from printing stderr unless errors...
        if  len(out.stderr) > 0:
            #print("STDERR: ", out.stderr, file=sys.stderr)
            print("STDERR: ", out.stderr, flush=True)
            breakpoint();
        if self.exp.args.verbose:
            print(out)
        return out

    def check_results(self, x):
        out = self.run_cmd(f"./bin/macduff run/corrected.jpg run/check.jpg --restore {self.exp.csv}")
        self.exp.cur_x   = x
        self.exp.cur_res = self.look_thru_cmd_output(out)
        # store check image with attributes
        #self.exp.counter += 1
        self.exp.check_image = f"run/check/check_{self.exp.tag}_run{self.exp.counter}.jpg"
        data = self.exp.get_image_data_for_json()
        comment = dumps(data)
        #comment = f""" args:{self.exp.args},x:{self.exp.cur_x},res:{self.exp.cur_res},error:{self.exp.cur_error} """
        cmd = f"""/bin/rm -f {self.exp.check_image} ;  exiftool -o {self.exp.check_image} -comment="{comment}" run/check.jpg"""
        self.run_cmd(cmd)
        
        self.record_if_best_results()
        self.print_status()
        #pu.db
        return self.exp.cur_res

    def look_thru_cmd_output(self, out):
        # grey_average =  62.1
        self.exp.cur_error = {'greyFC_average': 5000, 'deltaE_average': 5000, 'deltaC_average': 5000}
        m = re.findall('(\w+_average) *= *([0-9]*[.,]{0,1}[0-9]*)',str(out.stdout))
        for thing,value in m:
            self.exp.cur_error[thing] = float(value)

        # eg for white balance test ignore color metric .. focus on grey
        cur_res = 0
        for thing,value in self.exp.cur_error.items():
            cur_res += self.exp.cost_multiplier[thing] * value 
        self.exp.cur_res = cur_res
        return cur_res

    def record_if_best_results(self):
        if self.exp.min_res is None:
            self.exp.min_res = self.exp.cur_res
        self.exp.min_res = min(self.exp.min_res,self.exp.cur_res)

        # record this state if res is optimal
        if math.isclose(self.exp.cur_res, self.exp.min_res, rel_tol=1e-3):
            self.exp.min_x = self.exp.cur_x
            self.exp.min_error = self.exp.cur_error
            if self.exp.blacklevelp is not None:
                self.exp.min_blacklevelp = self.exp.blacklevelp
            if self.exp.whitelevelp is not None:
                self.exp.min_whitelevelp = self.exp.whitelevelp
            # save results to file for later runs
            self.exp.save_best()



    def print_status(self):

        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'

        text = self.text

        # first define all the text
        text['title'] = f"op={self.exp.operation} method={self.exp.method}"
        text['run'] = f"run{self.exp.counter}"

        text['cur_error'] =  "(E,C,G) = "
        text['min_error'] =  "(E,C,G) = "

        i = 0
        for type in ['deltaE_average', 'deltaC_average', 'greyFC_average']:
                text['cur_error'] +=  f"{self.exp.cost_multiplier[type]} X {self.exp.cur_error[type]:4.1f} " 
                text['min_error'] +=  f"{self.exp.cost_multiplier[type]} X {self.exp.min_error[type]:4.1f} " 
                if i == 2:
                    text['cur_error'] +=  " = "
                    text['min_error'] +=  " = "
                else:
                    text['cur_error'] +=  " + "
                    text['min_error'] +=  " + "
                i += 1;
        
        
        text['cur_error'] +=  f"{self.exp.cur_res:5.2f}"
        text['min_error'] +=  f"{self.exp.min_res:5.2f}"

        if self.exp.blacklevelp is not None:
            text['cur_level'] = f"(black,white) = ({self.exp.blacklevelp:.2f}%,{self.exp.whitelevelp:.2f}%)"
            text['min_level'] = f"(black,white) = ({self.exp.min_blacklevelp:.2f}%,{self.exp.min_whitelevelp:.2f}%)"

        text['cur_x'] = np.array2string(self.exp.cur_x,    
            max_line_width=1000,precision=3,separator=",",floatmode='fixed',sign=' ', formatter={'float_kind':lambda x: "% 4.3f" % x})
        text['min_x'] = np.array2string(self.exp.min_x,
            max_line_width=1000,precision=3,separator=",",floatmode='fixed',sign=' ', formatter={'float_kind':lambda x: "% 4.3f" % x})

        # needed for bug in rich table
        text['cur_x'] = text['cur_x'].strip(']').strip('[')
        text['min_x'] = text['min_x'].strip(']').strip('[')

        self.mark_diff_in_key('cur_error','last_error')
        self.mark_diff_in_key('cur_x','last_x')

        # ok we're now ready to define the table
        table = Table(title=text['title'], show_lines=True, show_header=False, )
        #table.add_column("current", style="cyan", justify="center")
        #table.add_column("best", style="yellow", justify="center")

        if self.exp.blacklevelp is not None:
            table.add_row(text['run'], text['cur_error'], text['cur_level'], text['cur_x'])
            table.add_row("best"     , text['min_error'], text['min_level'], text['min_x'])
        else:
            table.add_row(text['run'], text['cur_error'], text['cur_x'])
            table.add_row("best"     , text['min_error'], text['min_x'])

        self.exp.console.print(table)
        #self.restore_best()
        

    def mark_diff_in_key(self,cur_key, last_key):
        text = self.text
        if last_key not in text:
            text[last_key] = text[cur_key];
        str = mark_difference(text[cur_key], text[last_key])
        self.text[last_key] = text[cur_key];
        text[cur_key] = str
        #pdb.set_trace() 

# end class Operation
            


class SetupGenProfiles:
    def __init__(self, exp):
        self.exp = exp
        # store profile generator function in exp object, don't worry if no definition
        self.exp.gen_profile = getattr(self, self.exp.operation, None)

    def rt_rgb_curves(self, x):
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

    def rt_channel_mixer(self,x):

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


    def rt_color_toning(self, x):

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

    def rt_luminance_curve(self,x):

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


    def rt_hsv_equalizer(self,x):

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

#
# magic to get src/dest, ie: 
#   white balance is run on original 
#   color correction is run on white balance
#   blacklevel/whitelevel ccm is run on wb assuming bl
#
def get_destination_from_operation(src, operation):

    if operation.startswith("dng_"):
        suffix = operation.lstrip('dng')
        basename = os.path.basename(src).rsplit('.',1)[0]
        dst = f"inputs/dst/{basename}{suffix}.dng"
    else:
        dst = None
        
    return dst


def copy2(src,dst):
    dirname = os.path.dirname(dst)
    os.makedirs(dirname, exist_ok = True) 
    shutil.copy2(src, dst)

def mark_difference(str1,str2):
    # mark diff
    str = ""
    for ch1,ch2 in zip(str1,str2):
        if ch1 == ch2:
            str += ch1
        else:
            str += f"[bold magenta]{ch1}[/bold magenta]"
    return str


def get_csv_from_src(src):
    # inputs/img00006_G000E0400_wb_ccm.dng -> inputs/img00006_G000E0400_wb_ccm.csv
    base = src.rsplit('.',1)[0]
    csv = base + ".csv"
    if not os.path.isfile(csv):
        print(f"ERROR: can't find csv file {csv} \n");
        quit()
    return csv
    

def get_args_string(x):
    # preprocess x 
    #if profile is None:
    #    y = x
    #else:
    #   y = profile(x)
    args = "\"";
    for i in range(x.size):
        args += str(x[i]) + " ";
    args = args.strip()
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
def constraint_positive(x):
    if (x < 0).any():
        return -1
    else:        
        return 1

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
    return 1


def write_rt_profile(out):
    fp = open('profile.pp3', 'w')
    fp.write(out)
    fp.close()


def clamp(n, minn, maxn):
    return min(max(n, minn), maxn)
