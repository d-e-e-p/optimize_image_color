#!/usr/bin/python3
"""
 helper to optimize_image_color.py
 classes
    Exp :   main entry point object called by optimization routines
        Res :   store results
        Run :   inputs and stuff to execute runs
        Out :   output filenames

"""

import io, os, sys, re, math, copy, shutil, subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from collections import defaultdict
import pudb

import rich
from rich.console import Console
from rich.table import Table

from json_tricks import dumps, loads

# results
class Res:
    def __init__(self):
        self.x   = None
        self.blacklevelp = None
        self.whitelevelp = None
        self.cost_multiplier = {'greyFC_average': 0, 'deltaE_average': 1, 'deltaC_average': 0} 
        self.error = {'greyFC_average': 5000, 'deltaE_average': 5000, 'deltaC_average': 5000}
        self.cost = None
# inputs
class Run:
    def __init__(self, args, operation):
        self.args  = args
        self.operation = operation
        self.counter = 0
        self.check_image = None
        self.profile = None
        self.exifcmd = None
        self.text = {}
        self.series = []
        self.min_counter_series = []

        if self.args.src is not None:
            self.src = self.args.src
        else:
            self.src = "inputs/images/img00006_G000E0400.dng"
        # assume csv is adjacent to source image
        self.csv = get_csv_from_src(self.src)
        # mark this run with unique name
        if self.args.tag is not None:
            self.tag = self.args.tag
        else:
            basename = os.path.basename(self.src).rsplit('.',1)[0]
            self.tag = f"{basename}_{self.operation}"

# outputs
class Out:
    def __init__(self, src, operation ):
        self.basename = get_basename_from_source_file(src)
        dirname  = f"results/{self.basename}"
        os.makedirs(dirname, exist_ok = True)

        self.json          = f"{dirname}/{operation}.json"
        self.check_image   = f"{dirname}/{operation}.jpg"
        self.cost_plotfile = f"{dirname}/{operation}_cost.png"
        self.opt_plotfile  = f"{dirname}/{operation}_opt.png"

        if operation.startswith("dng_"):
            self.exif = f"{dirname}/{operation}.exif"
            self.dng  = f"{dirname}/{operation}.dng"
            self.csv  = f"{dirname}/{operation}.csv"
        else:
            self.exif = None
            self.dng  = None
            self.csv  = None

# experiment execution
class Exp:
    def __del__(self):
        self.plot_status(final=True)

    def __init__(self, args, operation):

        # optimization vars init by SetupVars and used later by SetupCallback

        self.method = None  #   method unsed for optimization
        self.method = None  #   method unsed for optimization
        self.x0     = None  #   current trial input
        self.fun    = None  #   function to evaluate x0 and return cost
        self.bounds = None  #   limits on x0
        self.constraints = () #  constrins on x0, can't be None!

        self.parallel_coordinates_series = None

        terminal = shutil.get_terminal_size((1024, 40))
        self.console = Console(width=terminal.columns, force_terminal=True)

        # shared data struct 
        self.run = Run(args, operation)
        self.out = Out(self.run.src, operation)
        self.cur_res = Res() 
        self.min_res = Res() 

        print(f"src: {self.run.src}") 
        print(f"dst: {self.out.dng}") 
        print(f"default multipliers : {self.cur_res.cost_multiplier}")

        # ok do the main init setup in 3 easy steps: 
        SetupVars(self)         # variables for each operation
        SetupCallback(self)     # define cost function
        SetupGenProfiles(self)  # profiles for running some functions 

        # resore best results
        self.restore_best()
        
        if self.run.args.verbose:
            print(f"Init Exp with tag: {self.run.tag} ")

    #
    # save best results to json file
    #
    def save_best(self):

        self.run.min_counter_series.append([self.run.counter, self.cur_res.cost])

        # save the error image
        copy2(self.run.check_image, self.out.check_image)

        if self.out.dng is not None:
            src_in_dng = "run/corrected.dng"
            copy2(src_in_dng, self.out.dng)
            if self.out.exif is not None:
                self.run_cmd(f"exiftool -args {self.out.dng} > {self.out.exif}")

        if self.out.csv is not None:
            copy2(self.run.csv, self.out.csv)
            
        # save the best results x and min_res values
        data = self.get_image_data_for_json()
        json = dumps(data, indent=4)

        fp = open(self.out.json , 'w')
        fp.write(json)
        fp.close()

    def get_image_data_for_json(self):

        save_fields = """
            run
            out
            min_res
        """.split()

        # look thru object and save values
        data = {}
        for s in save_fields:
            data[s] = getattr(self, s, None) 
        return data

    #
    # restore results if exists
    #
    def restore_best(self):

        if not os.path.isfile(self.out.json): 
            return

        fp = open(self.out.json, 'r')
        json = fp.read()
        fp.close()

        data = loads(json)
        # support old style and new style json
        if 'min_x' in data.keys():
            self.x0 = data['min_x']
        elif 'min_res' in data.keys():
            min_res = data['min_res']
            self.x0 = min_res.x
        else:
            print(f"ERROR: could not load results from data = {data}")
            breakpoint();
            

        print(f"restored results from file {self.out.json} :")
        print(f"x0 = {self.x0}")

    def whoami(self, from_mod ):
        print("hello {0}. I am instance {1}".format(from_mod, self))

    def run_exifcmd(self, x, exifcmd):

        # first store the cmd for output
        self.run.exifcmd = exifcmd

        self.run_cmd(f"/bin/rm -f run/corrected.dng ; {exifcmd}")
        self.run_cmd("/usr/local/bin/rawtherapee-cli -Y -o run/corrected.jpg -c run/corrected.dng")
        res = self.check_results(x)
        return res


    def run_cmd(self, cmd):
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # TODO: stop rt from printing stderr unless errors...
        if  len(out.stderr) > 0:
            #print("STDERR: ", out.stderr, file=sys.stderr)
            print("STDERR: ", out.stderr, flush=True)
            breakpoint();
        if self.run.args.verbose:
            print(out)
        return out

    def check_results(self, x):

        # record x
        self.cur_res.x = x

        # setup output dirs
        os.makedirs("run/check", exist_ok = True) 

        out = self.run_cmd(f"./bin/macduff run/corrected.jpg run/check.jpg --restore {self.run.csv}")
        self.look_thru_cmd_output(out)

        # store check image with exif attributes
        self.run.counter += 1
        if self.run.args.save_images:
            self.run.check_image = f"run/check/check_{self.run.tag}_run{self.run.counter}.jpg"
        else:
            self.run.check_image = f"run/check/check_{self.run.tag}.jpg"

        data = self.get_image_data_for_json()
        comment = dumps(data)
        cmd = f"""/bin/rm -f {self.run.check_image} ;  exiftool -o {self.run.check_image} -comment='{comment}' run/check.jpg"""
        self.run_cmd(cmd)
        
        self.record_if_best_results()
        self.append_vector_and_results()
        self.print_status()
        self.plot_status()
        return self.cur_res.cost

    def look_thru_cmd_output(self, out):
        # grey_average =  62.1
        self.cur_res.error = {'greyFC_average': 5000, 'deltaE_average': 5000, 'deltaC_average': 5000}
        m = re.findall('(\w+_average) *= *([0-9]*[.,]{0,1}[0-9]*)',str(out.stdout))
        for thing,value in m:
            self.cur_res.error[thing] = float(value)
        if self.cur_res.error['greyFC_average'] == 5000:
            print(f"ERROR: greyFC_average = 5000 with out={out}", flush=True)
            pu.db

        # eg for white balance test ignore color metric .. focus on grey
        cost = 0
        for thing,value in self.cur_res.error.items():
            cost += self.cur_res.cost_multiplier[thing] * value 
        self.cur_res.cost = cost

    def record_if_best_results(self):
        if self.min_res.cost is None:
            self.min_res = copy.deepcopy(self.cur_res)
        else:
            if self.cur_res.cost < self.min_res.cost:
                # record and save results to file for posterity
                self.min_res = copy.deepcopy(self.cur_res)
                self.save_best()

    def append_vector_and_results(self):

        self.run.series.append(self.cur_res.cost)
        data_line = np.append(self.cur_res.x,  self.cur_res.cost)
        if self.parallel_coordinates_series is None:
            self.parallel_coordinates_series = data_line
        else:
            # look for numpy.c ... a lot to unpack in one line
            #self.parallel_coordinates_series = \
            #    np.c_['1', self.parallel_coordinates_series, np.append(self.x0, self.cur_res.cost)]
            self.parallel_coordinates_series = np.append(self.parallel_coordinates_series, data_line)
        
            #print(f" x0ser = {self.parallel_coordinates_series}")

    def setup_plots(self):
        plt.style.use('dark_background')

        # figure1
        self.fig1 = plt.figure(1)
        self.fig1.set_size_inches(19.20, 10.24)

        # (nrows, ncols, and index)
        self.ax1 = self.fig1.add_subplot(2, 1, 1)
        self.ax2 = self.fig1.add_subplot(2, 1, 2)

        self.ax1.grid(True, linestyle='dotted')
        self.ax2.axis('off')
        print(f"now gcf = {plt.gcf()}")

        # figure2
        self.fig2 = plt.figure(2)
        self.fig2.set_size_inches(19.20, 10.24)

        # (nrows, ncols, and index)
        n = self.x0.size
        self.axes = []
        nrows = 1
        ncols = n
        for index in range(1,n):
            self.axes.append( self.fig2.add_subplot(nrows, ncols, index) )

        self.fig2.subplots_adjust(wspace=0)


    def plot_status(self, final=False):

        # skip a bunch of images before updating plot
        num_of_images_to_skip = 1
        #num_of_images_to_skip = 25
        if not final and (self.run.counter % num_of_images_to_skip != 0):
            return

        # if plot is init to 640x480, we need to reinit
        # this happens once at start and another time at final
        #plt.figure(1)
        if plt.gcf().bbox_inches.width == 6.4 :
            print(f"gcf = {plt.gcf()}")
            self.setup_plots()

        self.plot_convergence()
        self.plot_optimization()

        if final:
            plt.close('all')

    # results vs iterations
    def plot_convergence(self):

        self.fig1.suptitle(self.run.text['title'], color='green', fontsize=28, fontweight='bold')

        # first axis
        #self.ax1.subplots_adjust(bottom=0.5)
        self.ax1.set(xlabel='iteration', ylabel= 'cost')
        self.ax1.plot(self.run.series)

        # annotate first plot with pointer to min values
        for x,y in self.run.min_counter_series:
            self.ax1.annotate(f"min {y:.2f}", 
                xy=(x-1, y), xycoords='data',
                xytext=(0, -100), textcoords="offset points",
                arrowprops=dict(arrowstyle="simple", connectionstyle="arc3"),)

        # second axis
        textstr = self.run.table_str
        props = dict(boxstyle='round', facecolor='DarkSlateGray', alpha=0.9, )
        self.ax2.text(0.5,0.2, textstr, ha="center", va="center",fontsize=10, fontfamily='monospace', wrap=False, bbox=props)

        self.fig1.savefig(self.out.cost_plotfile, dpi=300)
        copy2(self.out.cost_plotfile, "run/series_cost.png")

    # parameteters
    def plot_optimization(self):

        n = self.x0.size + 1
        self.fig2.suptitle(self.run.text['title'], color='green', fontsize=28, fontweight='bold')
        data = self.parallel_coordinates_series.reshape(-1,n)
        df = pd.DataFrame(data=data)

        # too small ...
        if df.index.size < 3:
            return

        create_parallel_coordinates_plot(self.fig2, self.axes, n, df)
        self.fig2.savefig(self.out.opt_plotfile, dpi=300)
        copy2(self.out.opt_plotfile, "run/series_opt.png")

        """
        for i in range(n):
            ax = self.axes[i]
            ax.set(xlabel=f"x{i}")
            data = self.plot_parallel_get_data(i)
            lines = ax.plot(data, 'k')

            # configure axes
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_position(('outward', 5))
            ax.spines['bottom'].set_visible(True)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('none')

            # set limit to show only single line segment
            ax.set_xlim((i, i+1))
            ax.set_xticks([i])
            #ax.set_xticklabels([coordinates[i]])

            # set the scale
            #ax.set_yticks([0, 1])
            #ax.set_yticklabels([vmin[i], vmax[i]])


        """



    def print_status(self):

        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'

        # to make it easier to type..need to remember text over iterations for marking
        text = self.run.text
        cur_res = self.cur_res
        min_res = self.min_res

        # first define all the text
        text['title'] = f"input={self.out.basename} op={self.run.operation} method={self.method}"
        text['run'] = f"run{self.run.counter}"

        text['cur_error'] =  "(E,C,G) = "
        text['min_error'] =  "(E,C,G) = "


        i = 0
        for type in ['deltaE_average', 'deltaC_average', 'greyFC_average']:
                text['cur_error'] +=  f"{cur_res.cost_multiplier[type]} X {cur_res.error[type]:4.1f} " 
                text['min_error'] +=  f"{min_res.cost_multiplier[type]} X {min_res.error[type]:4.1f} " 
                if i == 2:
                    text['cur_error'] +=  " = "
                    text['min_error'] +=  " = "
                else:
                    text['cur_error'] +=  " + "
                    text['min_error'] +=  " + "
                i += 1;
        
        
        text['cur_error'] +=  f"{cur_res.cost:5.2f}"
        text['min_error'] +=  f"{min_res.cost:5.2f}"

        if cur_res.blacklevelp is not None:
            text['cur_level'] = f"(black,white) = ({cur_res.blacklevelp:.2f}%,{cur_res.whitelevelp:.2f}%)"
            text['min_level'] = f"(black,white) = ({min_res.blacklevelp:.2f}%,{min_res.whitelevelp:.2f}%)"

        text['cur_x'] = np.array2string(cur_res.x,    
            max_line_width=1000,precision=3,separator=",",floatmode='fixed',sign=' ', formatter={'float_kind':lambda x: "% 4.3f" % x})
        text['min_x'] = np.array2string(min_res.x,
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

        if cur_res.blacklevelp is not None:
            table.add_row(text['run'], text['cur_error'], text['cur_level'], text['cur_x'])
            table.add_row("best"     , text['min_error'], text['min_level'], text['min_x'])
        else:
            table.add_row(text['run'], text['cur_error'], text['cur_x'])
            table.add_row("best"     , text['min_error'], text['min_x'])

        self.console.print(table)
        self.run.table_str = self.render_table(table)

    def render_table(self, table) -> str:
        console = Console(width=256, file=io.StringIO(), color_system="standard")
        console.print(table)
        output = console.file.getvalue()
        output = strip_ansi_escape(output)
        return output


    def mark_diff_in_key(self,cur_key, last_key):
        text = self.run.text
        if last_key not in text:
            text[last_key] = text[cur_key];
        str = mark_difference(text[cur_key], text[last_key])
        self.run.text[last_key] = text[cur_key];
        text[cur_key] = str

# end class Exp

#
# SetupVars
#
class SetupVars:
    def __init__(self, exp):
        self.exp = exp
        # call the setup function based on operation variable, 
        # ie. if operation = dng_wb then run the dng_wb() function below
        func = getattr(self, self.exp.run.operation)
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
        self.exp.cur_res.cost_multiplier = {'greyFC_average': 0.9, 'deltaE_average': 0, 'deltaC_average': 0.1} 
        print(f"reset multipliers to : {self.exp.cur_res.cost_multiplier}")

    def dng_wb_bl(self):
        self.exp.x0 = np.array( [
                1, 1,       # wb 
                0, 0,       # bl, wl
        ])
        self.exp.bounds = [(0,2)] * self.exp.x0.size
        self.exp.constraints={"fun": constraint_positive, "type": "ineq"}
        self.exp.cur_res.cost_multiplier = {'greyFC_average': 0.9, 'deltaE_average': 0, 'deltaC_average': 0.1} 
        print(f"reset multipliers to : {self.exp.cur_res.cost_multiplier}")

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
        self.exp.cur_res.cost_multiplier = {'greyFC_average': 0.1, 'deltaE_average': 0.9, 'deltaC_average': 0.0} 
        print(f"reset multipliers to : {self.exp.cur_res.cost_multiplier}")

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
        self.exp.cur_res.cost_multiplier = {'greyFC_average': 0.0, 'deltaE_average': 1.0, 'deltaC_average': 0.0} 
        print(f"reset multipliers to : {self.exp.cur_res.cost_multiplier}")

    def eq_spline(self):

        # force inputs to be in increasing order in calling function
        self.exp.constraints={"fun": constraint_monotonic, "type": "ineq"}

        self.exp.x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
        ])

        self.exp.bounds = [(0.1,0.9)] * self.exp.x0.size

    def eq_root_polynomial(self):
        self.exp.x0 = np.array([
            1.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
            0.,  1,  0,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
            0.,  0,  1,  0.,  0,  0,  0.,  0,  0,  0.,  0,  0,  0,
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def eq_multi_ccm2(self):
        self.exp.x0 = np.array([
            1, 0, 0,  1, 0, 0, 
            0, 1, 0,  0, 1, 0, 
            0, 0, 1,  0, 0, 1, 
        ])
        self.exp.bounds = [(-2,2)] * self.exp.x0.size

    def eq_multi_ccm4(self):
        self.exp.x0 = np.array([
            1,0,0, 1,0,0, 1,0,0, 1,0,0,
            0,1,0, 0,1,0, 0,1,0, 0,1,0,
            0,0,1, 0,0,1, 0,0,1, 0,0,1,
        ])
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
        self.exp.constraints={"fun": constraint_monotonic, "type": "ineq"}
        self.exp.x0 = np.array([
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
            0.25,  0.5,  0.75,
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
        # just store function for later in Exp object
        self.exp.fun = getattr(self, self.exp.run.operation, None)
        if not self.exp.fun:
            if self.exp.run.operation.find("rt_") == 0 :
                self.exp.fun = getattr(self, 'rt_generic')
            elif self.exp.run.operation.find("eq_") == 0 :
                self.exp.fun = getattr(self, 'eq_generic')
            else:
                print(f"ERROR: handler not found for operation {self.exp.run.operation}")
                breakpoint();


    def dng_wb(self,x):

        # only allow R and B to vary holding G at 1
        r,g,b = (x[0], 1, x[1])

        #force r and b to be positive
        if r < 0: return 100 - 100 * r 
        if b < 0: return 100 - 100 * b 
            
        args = f"""\"{r} {g} {b}\" """
        
        res = self.exp.run_exifcmd(x, f"exiftool -AsShotNeutral={args} -o run/corrected.dng  {self.exp.run.src}")
        return res

    def dng_wb_ccm(self,x):

        # first 2 elements are for wb
        r,g,b = (x[0], 1, x[1])
        args_wb = f"""\"{r} {g} {b}\" """

        #force r and b to be positive
        if r < 0: return 100 - 100 * r 
        if b < 0: return 100 - 100 * b 

        # next 9 elements for ccm
        y = x[2:11:1] 
        args_ccm = get_args_string(y)

        res = self.exp.run_exifcmd(x, f"exiftool -AsShotNeutral={args_wb}  -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args_ccm} -ColorMatrix2={args_ccm} -o run/corrected.dng  {self.exp.run.src}")
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
        res = self.exp.run_exifcmd(x, f"exiftool -AsShotNeutral={args} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.run.src}")
        return res

    def dng_ccm (self,x):
        args = get_args_string(x)
        res = self.exp.run_exifcmd(x, f"exiftool -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args} -ColorMatrix2={args} -o run/corrected.dng  {self.exp.run.src}")
        return res

    def dng_ccm_bl(self,x):
        # extra conf at end for blacklevel
        y = x[0:9:1] # first 9 elements only
        args = get_args_string(y)
        self.set_blackwhite_level(x[9], x[10])

        res = self.exp.run_exifcmd(x, f"exiftool -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args} -ColorMatrix2={args} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.run.src}")
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

        res = self.exp.run_exifcmd(x, f"exiftool -AsShotNeutral={args_wb}  -ForwardMatrix1= -ForwardMatrix2= -ColorMatrix1={args_ccm} -ColorMatrix2={args_ccm} -IFD0:BlackLevel={self.exp.blacklevel} -IFD0:WhiteLevel={self.exp.whitelevel} -SubIFD:BlackLevelRepeatDim= -SubIFD:BlackLevel={self.exp.blacklevel} -SubIFD:WhiteLevel={self.exp.whitelevel}  -o run/corrected.dng  {self.exp.run.src}")
        return res

    def eq_spline(self,x):
        x = force_monotonic(x)
        return self.eq_generic(x)

    def rt_generic(self,x):
        self.exp.gen_profile(x)
        self.exp.run_cmd(f"/usr/local/bin/rawtherapee-cli -Y -p run/profile.pp3 -o run/corrected.jpg -c {self.exp.run.src}")
        return self.exp.check_results(x)

    def eq_generic(self,x):
        args = get_args_string(x)
        self.exp.run_cmd(f"./bin/rawtherapee-cli -Y -o run/corrected.jpg -e {self.exp.run.operation} -m {args} -c {self.exp.run.src}")
        return self.exp.check_results(x)

    def set_blackwhite_level(self, bl, wl):
        # assume 16bit images
        # TODO: make it work for 8-bit inputs
        # for eg2 Black=1024 White=15600
        bl = clamp(bl,0,2)
        wl = clamp(wl,0,2)
        sixteenbit = 2**16 - 1

        # bl=2 makes black=10% (instead of 0) while wl=2 makes white=20% (instead of 100%)
        self.exp.cur_res.blacklevelp = 5 * bl 
        self.exp.cur_res.whitelevelp = 100 - (40 * wl)
        self.exp.blacklevel  = int(sixteenbit * self.exp.cur_res.blacklevelp / 100.0)
        self.exp.whitelevel  = int(sixteenbit * self.exp.cur_res.whitelevelp / 100.0)

# end class SetupCallback



class SetupGenProfiles:
    def __init__(self, exp):
        self.exp = exp
        # store profile generator function in exp object, don't worry if no definition
        self.exp.gen_profile = getattr(self, self.exp.run.operation, None)

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

# from https://benalexkeen.com/parallel-coordinates-in-matplotlib/

def create_parallel_coordinates_plot(fig, axes, n, df):

    cost = 9

    hist, bin_edges = np.histogram(df[cost], bins=10 )
    cut_points = [ 0, bin_edges[1], bin_edges[3], bin_edges[7], bin_edges[-1], ]
    df[cost] = pd.cut(df[cost], cut_points)


    cols = list(range(0, 9))
    x = [i for i, _ in enumerate(cols)]
    colours = ['#2e8ad8', '#cd3785', '#c64c00', '#889a00']

    print(f"x={x}")
    for i, ax in enumerate(axes):
        print(f"i:{i}")
        print(f"i:{i} {x[i]} to {x[i+1]}")

    # create dict of categories: colours
    colours = {df[cost].cat.categories[i]: colours[i] for i, _ in enumerate(df[cost].cat.categories)}

    # Create (X-1) sublots along x axis
    #fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))
    for i, ax in enumerate(axes):
        ax.set(xlabel=f"x{i}")

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        if np.ptp(df[col]) > 0:
            df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    print(f"df3:{df}")
    # convert to list
    #if type(axes) is not list:
    #    axes = [axes]
    
    for i, ax in enumerate(axes):
        for idx in df.index:
            res_category = df.loc[idx, cost]
            #print(f"plot x={x} d={df.loc[idx, cols]} c={colours[res_category]}")
            ax.plot(x, df.loc[idx, cols], colours[res_category])
        ax.set_xlim([x[i], x[i+1]])


    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks-1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks-1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])


    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Add legend to plot
    ax.legend(
        [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in df[cost].cat.categories],
        df[cost].cat.categories,
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

    # Remove space between subplots
    fig.subplots_adjust(wspace=0)






#
#

# derive basename from either root of file or directory name if filename is generic dng_*
def get_basename_from_source_file(src):
    base = src.rsplit('.',1)[0]
    basename = os.path.basename(base)
    if basename.startswith("dng_"):
        basename = os.path.basename(os.path.dirname(src))
    return basename 
    
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
            str += f"[bold red]{ch1}[/bold red]"
    return str


def get_csv_from_src(src):
    # inputs/img00006_G000E0400_wb_ccm.dng -> inputs/img00006_G000E0400_wb_ccm.csv
    base = src.rsplit('.',1)[0]
    csv = base + ".csv"
    if not os.path.isfile(csv):
        print(f"ERROR: can't find csv file {csv} \n");
        quit()
    return csv
    
def get_csv_from_dst(dst):
    base = dst.rsplit('.',1)[0]
    csv = base + ".csv"
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

# from https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
def strip_ansi_escape(sin):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    sout = ansi_escape.sub('', sin)
    return sout

def write_rt_profile(out):
    fp = open('run/profile.pp3', 'w')
    fp.write(out)
    fp.close()


def clamp(n, minn, maxn):
    return min(max(n, minn), maxn)



