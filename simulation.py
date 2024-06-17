import cupy as cp
from cupyx.scipy import ndimage
import os
import yaml
import pandas as pd
import numpy as np
import io
import cv2
import time
import cv2
import functools 
import colormap
import ast
import numpy as np
import scipy.interpolate
import yaml
from matplotlib import pyplot as plt
import tqdm
import subprocess
#import ffmpeg

# load parameters
parameter_file = "parameters.yml"

parameters = {}
with open(parameter_file, "r") as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)

def rgb_to_hsv(rgb):
    xp = cp.get_array_module(rgb)
    hsv = xp.zeros(rgb.shape)
    maxc = xp.max(rgb,axis=2)
    minc = xp.min(rgb,axis=2)
    rangec = (maxc-minc)
    hsv[:,:,2] = maxc
    hsv[:,:,1] = rangec / maxc * 255
    rc = (maxc-rgb[:,:,0]) / rangec
    gc = (maxc-rgb[:,:,1]) / rangec
    bc = (maxc-rgb[:,:,2]) / rangec
    h = hsv[:,:,0]
    h = (4.0+gc-rc) * (1- xp.equal(rgb[:,:,0],maxc) ) *  (1- xp.equal(rgb[:,:,1],maxc) ) + xp.equal(rgb[:,:,0],maxc) * (bc-gc) + xp.equal(rgb[:,:,1],maxc) * (2.0+rc-bc)
    h = ((h/6.0) % 1.0) * 180
    hsv[:,:,0] = h
    return xp.rint(hsv).astype('uint8')

def hsv_to_rgb(hsv):
    xp = cp.get_array_module(hsv)
    h = hsv[:,:,0]/180
    s = hsv[:,:,1]/255
    v = hsv[:,:,2]
    i  = xp.trunc(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    ones = xp.ones(i.shape)
    rgbT = xp.equal(i,0*ones) * xp.array([v, t, p]) + xp.equal(i,1*ones) * xp.array([q, v, p]) +xp.equal(i,2*ones) * xp.array([p, v, t]) +xp.equal(i,3*ones) * xp.array([p, q, v]) +xp.equal(i,4*ones) * xp.array([t, p, v]) +xp.equal(i,5*ones) * xp.array([v, p, q])
    rgb = xp.moveaxis(rgbT,0,2)
    return xp.rint(rgb).astype('uint8')

def build_checkerboard(w, h) :
    N= 8
    re = np.r_[ w*(N*[0]+N*[1]) ]              # even-numbered rows
    ro = np.r_[ w*(N*[1]+N*[0]) ]              # odd-numbered rows
    return np.row_stack(h*(N*(re,)+N*(ro,)))[:w,:h]

class Simulation_data:
    def __init__(self):

        # Constant fields
        self.fertility_map = cp.array([])
        self.population_diffusivity_map = cp.array([])   

        # Time dependant fields 
        self.population = cp.array([])
        self.technological_level = cp.array([])
        # self.total_culture = np.zeros((num_rows, num_cols))
        # self.political_stage = np.zeros((num_rows, num_cols))
        # self.culture_vector = np.zeros((num_rows, num_cols,max_number_of_cultures))

        # Constants
        self.natural_growth = 0

        # Variables
        self.year = 0

        # Functions
        self.fertility_per_technology_level = pd.DataFrame()

    def set_max_population(self):
        self.Pmax = self.ha_per_px*self.fertility_per_year[self.fertility_per_year['year']==self.year].iloc[0]["max_population_per_ha"]*self.fertility_map

    def reset_simulation(self):
        # Load image
        map_path = parameters['geographics']['map_path']  # Path to your map image
        self.map_img = cv2.imread(map_path)
        self.map_img = cv2.cvtColor(self.map_img, cv2.COLOR_BGR2RGB)
        self.map_img_cp = cp.asarray(self.map_img)
        self.map_height, self.map_width, _ = self.map_img.shape

        # Load constants
        self.natural_growth = parameters['demographics']['natural_growth']
        self.ha_per_px = parameters['geographics']['ha_per_px']
        self.start_year = parameters['time']['start_year']
        self.end_year = parameters['time']['end_year']
        self.time_step = parameters['time']['time_step']

        # Compute geographical dependant coefficient for fertility and population_diffusivity
        geographics_params = [ {**geo_param, "RGB" : ast.literal_eval(geo_param['RGB']) } for geo_param in  parameters['geographics']['zones']]
        geographics_params = [ {**geo_param, "color" : geo_param['RGB'][0]+256*geo_param['RGB'][1]+(256^2)*geo_param['RGB'][2] } for geo_param in  geographics_params]

        color_values = [x['color'] for x in geographics_params]
        fertility_values = [x['Fertility'] for x in geographics_params]
        diffusivity_values = [x['population_diffusivity'] for x in geographics_params]

        fertility_polynomial_coefs = scipy.interpolate.lagrange(color_values, fertility_values).coef
        population_diffusivity_polynomial_coefs = scipy.interpolate.lagrange(color_values, diffusivity_values).coef


        X = self.map_img[:,:,0].astype(np.int32)+256*self.map_img[:,:,1].astype(np.int32)+(256^2)*self.map_img[:,:,2].astype(np.int32)
        fertility_map = np.zeros(X.shape)
        population_diffusivity_map = np.zeros(X.shape)

        for i in range(max(len(fertility_polynomial_coefs),len(fertility_polynomial_coefs))):
            fertility_map = np.multiply(fertility_map,X) + fertility_polynomial_coefs[i]
            population_diffusivity_map = np.multiply(population_diffusivity_map,X)  + population_diffusivity_polynomial_coefs[i]

        self.fertility_map = cp.asarray(fertility_map)
        self.population_diffusivity_map = cp.asarray(population_diffusivity_map)

        #self.fertility_map = ndimage.gaussian_filter(self.fertility_map,order=0,sigma=self.time_step)
        self.population_diffusivity_map = ndimage.gaussian_filter(self.population_diffusivity_map,order=0,sigma=self.time_step)

        # Functions
        self.fertility_per_technology_level = pd.DataFrame(parameters['fertility_per_technology_level'])
        years = sorted(list(dict.fromkeys(self.fertility_per_technology_level['year'].tolist()+list(range(self.start_year, self.end_year + 1 ,1)))))
        self.fertility_per_year=pd.merge(pd.DataFrame({ 'year' : years,}),self.fertility_per_technology_level[['year','max_population_per_ha']],on="year",how="left").interpolate(method='linear')
        self.fertility_per_year['max_population_per_ha'] *= float(parameters['demographics']['population_trim'])

        #initialisation
        self.year = self.start_year
        self.population = cp.asarray(np.zeros(self.map_img[:,:,0].shape))
        ## TODO parametrize population start
        
        self.set_max_population()
        self.population[800,800] =self.Pmax[800,800]
        self.population = ndimage.gaussian_filter(self.population,order=0,sigma=self.time_step)

        # display
        self.image_list = []
        self.checkerboard = cp.asarray(build_checkerboard(self.map_img.shape[0],self.map_img.shape[1]))[:self.map_img.shape[0],:self.map_img.shape[1]]

    def view_field(self,field_name):
        X = getattr(self, field_name)

        xp = cp.get_array_module(X)
        Y = X.copy()
        Ya = xp.abs(Y)

        match field_name:
            case "population":
                h = xp.zeros(Y.shape) # red
                # v = xp.power(xp.min(Ya,1),0.25) * self.fertility_map * 255
                # s = (1-xp.power(Ya/self.ha_per_px*2.5,0.25)) * 255
                v = xp.zeros(Y.shape)
                v[Y>1] = ((128 + 127 * xp.power(Ya/xp.max(self.Pmax),0.25)) * self.fertility_map)[Y>1] #Population is displayed in saturated red when it reaches Pmax
                v[(Y<1)*(Y>1e-5)] = 127 * (xp.power(Ya,0.25) * self.fertility_map)[(Y<1)*(Y>1e-5)] #Population is displayed in dark red when bellow 1
                population_max_bound = self.ha_per_px*self.fertility_per_year['max_population_per_ha'].max()
                s = (xp.ones(Y.shape) - xp.power(Ya/(population_max_bound),1)*self.checkerboard)  * 255 #Population is displayed as checkflag when it reach the max population in history
            case "diffusion" :
                h = xp.zeros(Y.shape) # red
                h[Y<0] = (180/360) * 255 # blue
                v = xp.power(Ya/xp.max(Ya),0.25) * 255
                s = xp.ones(Y.shape) * 255
            case _ :
                h = xp.zeros(Y.shape) # red
                h[Y<0] = 170 # blue
                v = xp.power(Ya/xp.max(Ya),0.25) * 255
                s = xp.ones(Y.shape) *255
        
        hsv = xp.moveaxis(xp.array([h, s, v]),0,2)
        rgb = hsv_to_rgb(hsv)
        overlay = xp.zeros(self.map_img.shape)
        overlay[:,:,0]= self.map_img_cp[:,:,0]* (1-v/255.0) + v/255.0 * rgb[:,:,0]
        overlay[:,:,1]= self.map_img_cp[:,:,1]* (1-v/255.0) + v/255.0 * rgb[:,:,1]
        overlay[:,:,2]= self.map_img_cp[:,:,2]* (1-v/255.0) + v/255.0 * rgb[:,:,2]

        img = cp.asnumpy(overlay).astype('uint8')
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,40)
        fontScale              = 1.5
        fontColor              = (255,255,255)
        thickness              = 5
        lineType               = 1

        time = self.year-self.start_year

        img = cv2.putText(img,"Year="+f"{time:04}"+ ", Max "+field_name+"="+"{:.1e}".format(np.sum(X)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        #cv2.imwrite("img/"+field_name+"_"+f"{time:04}"+".png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return img
    
    def view_fields(self):
        img_pop = self.view_field("population")
        img_dif = self.view_field("diffusion")
        images = cv2.hconcat([img_pop, img_dif])
        time = self.year-self.start_year
        image_name = "img/sim_"+f"{time:04}"+".png"
        cv2.imwrite(image_name,cv2.cvtColor(images, cv2.COLOR_RGB2BGR))
        self.image_list += [image_name]
    
    def plot_field(self,field_name):
        plt.imshow(self.view_field(field_name))
        plt.show()
   
    def iterate(self):
        self.set_max_population()
        dx = ndimage.gaussian_filter(self.population,order=[0,1],sigma=self.time_step)
        dy = ndimage.gaussian_filter(self.population,order=[1,0],sigma=self.time_step)
        dkdx = ndimage.gaussian_filter(self.population_diffusivity_map*dx,order=[0,1],sigma=self.time_step) #self.population_diffusivity_map*
        dkdy = ndimage.gaussian_filter(self.population_diffusivity_map*dy,order=[1,0],sigma=self.time_step) #self.population_diffusivity_map*
        self.diffusion = (dkdx + dkdy)*float(parameters['geographics']['population_diffusivity_trim'])
        dP = (self.population > 0)* self.natural_growth*self.population*(1-self.population/self.Pmax) + self.diffusion      
        self.population += dP*self.time_step
        self.population[self.population < -self.Pmax] = -self.Pmax[self.population < -self.Pmax]
        #self.population[self.population < 1e-5] = 0
        self.year += self.time_step

if __name__ == "__main__":
    #initialisation
    obj = Simulation_data()
    obj.reset_simulation()
    #obj.plot_field("population")

    #processing
    N = int((obj.end_year - obj.start_year) / obj.time_step)
    with tqdm.tqdm(total=N) as pbar:
        while obj.year < obj.end_year:
            obj.iterate()
            obj.view_fields()
            pbar.update(1)  

    #video generation
    basepath = os.path.dirname(__file__)

    image_path = os.path.join(basepath,'img')
    image_list_filename = os.path.join(image_path,'concat.txt')

    concat = open(image_list_filename, 'w')
    for file in obj.image_list:
        concat.write("file '" +file +"'\n")
    concat.close()

    os.chdir(basepath)
    video_name = os.path.join(image_path,"civilisation.mp4")
    command = ['ffmpeg','-f','concat','-r','25','-y','-i',image_list_filename,video_name]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())