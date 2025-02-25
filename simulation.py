import cupy as cp
from cupyx.scipy import ndimage
import os
import yaml
import pandas as pd
import numpy as np
import cv2
import ast
import numpy as np
import scipy.interpolate
import yaml
from matplotlib import pyplot as plt
import tqdm
import subprocess
import concurrent.futures

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
    """
    Convert an HSV image to RGB.

    Parameters:
    -----------
    hsv : cupy.ndarray
        Input HSV image with shape (height, width, 3) and values in [0,1].
        - hsv[:,:,0] : Hue (H), normalized between 0 and 1
        - hsv[:,:,1] : Saturation (S), between 0 and 1
        - hsv[:,:,2] : Value (V), between 0 and 1

    Returns:
    --------
    rgb : cupy.ndarray
        Output RGB image with shape (height, width, 3) and values in [0,1].
        - rgb[:,:,0] : Red (R)
        - rgb[:,:,1] : Green (G)
        - rgb[:,:,2] : Blue (B)
    """
    # Extract H, S, V channels
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    # Compute H_prime (normalized hue for the chromatic wheel)
    H_prime = H * 6
    # Compute region index (0 to 5)
    i = cp.floor(H_prime).astype(cp.int32) % 6
    # Fractional part of H_prime
    f = H_prime - cp.floor(H_prime)

    # Intermediate values
    p = V * (1 - S)
    q = V * (1 - S * f)
    t = V * (1 - S * (1 - f))

    # Stack choices for R, G, B into single CuPy arrays
    choices_R = cp.stack([V, q, p, p, t, V], axis=0)  # Shape: (6, height, width)
    choices_G = cp.stack([t, V, V, q, p, p], axis=0)  # Shape: (6, height, width)
    choices_B = cp.stack([p, p, t, V, V, q], axis=0)  # Shape: (6, height, width)

    # Select values for R, G, B based on region index i
    R = cp.choose(i, choices_R)
    G = cp.choose(i, choices_G)
    B = cp.choose(i, choices_B)

    # Stack R, G, B channels into RGB image
    rgb = cp.stack([R, G, B], axis=2)
    return rgb

def h1sv_to_rgb(h1,sv):
    xp = cp.get_array_module(sv)
    h6 = h1*(6/180)
    s = sv[:,:,0]/255
    v = sv[:,:,1]
    i  = np.trunc(h6)
    f = (h6) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    match i:
        case 0:
            rgbT = xp.array([v, t, p])
        case 1:
            rgbT = xp.array([q, v, p])
        case 2:
            rgbT = xp.array([p, v, t])
        case 3:
            rgbT = xp.array([p, q, v])
        case 4:
            rgbT = xp.array([t, p, v])
        case 5:
            rgbT = xp.array([v, p, q])
    rgb = xp.moveaxis(rgbT,0,2)
    return xp.rint(rgb).astype('uint8')

def build_checkerboard(w, h) :
    N= 8
    re = np.r_[ w*(N*[0]+N*[1]) ]              # even-numbered rows
    ro = np.r_[ w*(N*[1]+N*[0]) ]              # odd-numbered rows
    return np.row_stack(h*(N*(re,)+N*(ro,)))[:w,:h]

def store_images(image_table,image_name):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,40)
    fontScale              = 1.5
    fontColor              = (255,255,255)
    thickness              = 5
    lineType               = 1
        
    images = cv2.vconcat([cv2.hconcat([cv2.putText(image['img'],image['text'],bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType) for image in row]) for row in image_table])
    cv2.imwrite(image_name,cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

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
        geographics_params = [ {**geo_param, "color" : geo_param['RGB'][0]+256*geo_param['RGB'][1]+(256**2)*geo_param['RGB'][2] } for geo_param in  geographics_params]

        color_values = [x['color'] for x in geographics_params]
        fertility_values = [x['Fertility'] for x in geographics_params]
        diffusivity_values = [x['population_diffusivity'] for x in geographics_params]

        fertility_polynomial_coefs = scipy.interpolate.lagrange(color_values, fertility_values).coef
        population_diffusivity_polynomial_coefs = scipy.interpolate.lagrange(color_values, diffusivity_values).coef


        X = self.map_img[:,:,0].astype(np.int32)+256*self.map_img[:,:,1].astype(np.int32)+(256**2)*self.map_img[:,:,2].astype(np.int32)
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
        self.culture = 0.02 * cp.random.randn(*(self.map_img[:,:,0].shape +(2,)))#La culture est dimension 2
        ## TODO parametrize population start
        
        self.set_max_population()
        for location in parameters['initalisation']:
            self.population[location["i"],location["j"]] = location["population"] 
        self.population = ndimage.gaussian_filter(self.population,order=0,sigma=self.time_step)

        # display
        self.image_list = []
        self.checkerboard = cp.asarray(build_checkerboard(self.map_img.shape[0],self.map_img.shape[1]))[:self.map_img.shape[0],:self.map_img.shape[1]]
        self.Pool = concurrent.futures.ThreadPoolExecutor()
        self.display_tasks = []

    def view_field(self,field_name):
        X = getattr(self, field_name)

        xp = cp.get_array_module(X)
        Y = X.copy()
        Ya = xp.abs(Y)

        match field_name:
            case "population":
                h_red = 0 # red
                # v = xp.power(xp.min(Ya,1),0.25) * self.fertility_map * 255
                # s = (1-xp.power(Ya/self.ha_per_px*2.5,0.25)) * 255
                v = xp.zeros(Y.shape)
                v[Y>1] = ((128 + 127 * xp.power(Ya/xp.max(self.Pmax),0.25)) * self.fertility_map)[Y>1] #Population is displayed in saturated red when it reaches Pmax
                v[(Y<1)*(Y>1e-5)] = 127 * (xp.power(Ya,0.25) * self.fertility_map)[(Y<1)*(Y>1e-5)] #Population is displayed in dark red when bellow 1
                population_max_bound = self.ha_per_px*self.fertility_per_year['max_population_per_ha'].max()
                #s = (xp.ones(Y.shape) - xp.power(Ya/(population_max_bound),1)*self.checkerboard)  * 255 #Population is displayed as checkflag when it reach the max population in history
                s = xp.zeros(Y.shape) * 255
                sv = xp.moveaxis(xp.array([s, v]),0,2)
                rgb = h1sv_to_rgb(h_red,sv)

                displayed_value = xp.sum(X)
            case "diffusion" :
                h_red = 0
                h_blue = (180/360) * 255
                v = xp.power(Ya/xp.max(Ya),0.25) * 255
                s = xp.ones(Y.shape) * 255

                v_red = v.copy()
                v_red[Y<0] = (v*0)[Y<0]
                v_blue = v.copy()
                v_blue[Y>0] = (v*0)[Y>0]
                rgb = h1sv_to_rgb(h_red,xp.moveaxis(xp.array([s, v_red]),0,2)) + h1sv_to_rgb(h_blue,xp.moveaxis(xp.array([s, v_blue]),0,2))
                displayed_value = xp.sum(xp.abs(X))

            case "culture" :
                Y = self.population.copy()
                v = xp.zeros(Y.shape)
                v[Y>1] = ((128 + 127 * xp.power(Y/xp.max(self.Pmax),0.25)) * self.fertility_map)[Y>1] #Population is displayed in saturated red when it reaches Pmax
                v[(Y<1)*(Y>1e-5)] = 127 * (xp.power(Y,0.25) * self.fertility_map)[(Y<1)*(Y>1e-5)] #Population is displayed in dark red when bellow 1
                population_max_bound = self.ha_per_px*self.fertility_per_year['max_population_per_ha'].max()
                a, b = self.culture[:,:,0], self.culture[:,:,1]
                # Calculer la saturation comme la distance euclidienne au centre 
                hue = (xp.arctan2(b, a) / (2 * xp.pi)) % 1  # Normalisation entre 0 et 1
                saturation = xp.sqrt(a**2 + b**2) # Norme L2 normalisée
                #saturation /= xp.max(saturation)  # Normalisation entre
                value = v.copy()/255.0
                hsv = xp.stack([hue, saturation, value], axis=2)
                rgb = hsv_to_rgb(hsv)*255  
                culture_L2 = xp.sqrt(xp.sum(xp.square(self.culture),axis=2))
                displayed_value = xp.max(culture_L2)/255           
            
            case _ :
                h = 0 # red
                #h[Y<0] = 170 # blue
                v = xp.power(Ya/xp.max(Ya),0.25) * 255
                s = xp.ones(Y.shape) *255
                sv = xp.moveaxis(xp.array([s, v]),0,2)
                rgb = h1sv_to_rgb(h,sv)

        overlay = xp.zeros(self.map_img.shape)
        overlay[:,:,0]= self.map_img_cp[:,:,0]* (1-v/255.0) + v/255.0 * rgb[:,:,0]
        overlay[:,:,1]= self.map_img_cp[:,:,1]* (1-v/255.0) + v/255.0 * rgb[:,:,1]
        overlay[:,:,2]= self.map_img_cp[:,:,2]* (1-v/255.0) + v/255.0 * rgb[:,:,2]

        img = cp.asnumpy(overlay.astype('uint8'))
        time = self.year-self.start_year

        text = "Year="+f"{time:04}"+ ", Max "+field_name+"="+"{:.1e}".format(displayed_value)
        return text,img

    def view_fields(self):
        #text_pop,img_pop = self.view_field("population")
        #text_dif,img_dif = self.view_field("diffusion")
        text_dif,img_dif = self.view_field("culture")
        time = self.year-self.start_year
        image_name = "img/sim_"+f"{time:04}"+".png"
        self.image_list += [image_name]
        #self.display_tasks.append(self.Pool.submit(store_images,[[{'img' : img_pop, 'text' : text_pop},{'img' : img_dif,'text' : text_dif}]], image_name))
        self.display_tasks.append(self.Pool.submit(store_images,[[{'img' : img_dif,'text' : text_dif}]], image_name))
    
    def plot_field(self,field_name):
        plt.imshow(self.view_field(field_name)[1])
        plt.show()
   
    def iterate(self):
        xp = cp.get_array_module(self.population)
        # Précharger les paramètres constants sur GPU pour éviter les conversions implicites
        diffusivity_trim = xp.asarray(float(parameters['geographics']['population_diffusivity_trim']))
        natural_growth = xp.asarray(self.natural_growth)
        time_step = xp.asarray(self.time_step)
        culture_limit = xp.asarray(1.0)  # Constante sur GPU
        divergence_coeff = xp.asarray(parameters['culture']['divergence_coefficient'])
        
        # S'assurer que tous les attributs sont déjà des tableaux CuPy
        population = self.population
        Pmax = self.Pmax
        diffusivity_map = self.population_diffusivity_map
        culture = self.culture

        # 1. Calculs de diffusion de population
        # Fusionner les filtres gaussiens en une seule passe si possible (gain limité ici à cause des ordres)
        dPdx = ndimage.gaussian_filter(population, order=[0, 1], sigma=time_step)
        dPdy = ndimage.gaussian_filter(population, order=[1, 0], sigma=time_step)
        
        # Pré-calculer le produit une seule fois et réutiliser
        diffusivity_times_dPdx = diffusivity_map * dPdx
        diffusivity_times_dPdy = diffusivity_map * dPdy
        
        dPdx2 = ndimage.gaussian_filter(diffusivity_times_dPdx, order=[0, 1], sigma=time_step)
        dPdy2 = ndimage.gaussian_filter(diffusivity_times_dPdy, order=[1, 0], sigma=time_step)
        
        # Calcul vectorisé de la diffusion
        population_diffusion = (dPdx2 + dPdy2) * diffusivity_trim
        
        # Calcul de dP avec vectorisation complète
        growth_term = xp.where(population > 0, natural_growth * population * (1 - population / Pmax), 0)
        dP = growth_term + population_diffusion
        
        # Mise à jour de la population avec bornes
        population += dP * time_step
        population = xp.clip(population, -Pmax, None)  # Remplace population < -Pmax
        
        # Option : seuillage à 1e-5 (décommenter si nécessaire)
        # population = xp.where(population < 1e-5, 0, population)
        
        self.year += time_step

        # 2. Calculs de diffusion de culture
        sigma_culture = 5 * time_step
        sigma_culture_3d = [sigma_culture, sigma_culture, 0]  # Réutilisation
        
        dCdx = ndimage.gaussian_filter(culture, order=[0, 1, 0], sigma=sigma_culture_3d)
        dCdy = ndimage.gaussian_filter(culture, order=[1, 0, 0], sigma=sigma_culture_3d)
        
        # Pré-calculer le facteur commun et ajouter la dimension sans copie excessive
        pop_diffusivity_weight = (diffusivity_map * population)[:, :, xp.newaxis]
        dCdx_weighted = pop_diffusivity_weight * dCdx
        dCdy_weighted = pop_diffusivity_weight * dCdy
        
        dCdx2 = ndimage.gaussian_filter(dCdx_weighted, order=[0, 1, 0], sigma=sigma_culture_3d)
        dCdy2 = ndimage.gaussian_filter(dCdy_weighted, order=[1, 0, 0], sigma=sigma_culture_3d)
        
        # Calcul de la diffusion culturelle
        culture_diffusion = (dCdx2 + dCdy2) * diffusivity_trim / (xp.mean(population) + 1)
        
        # Calcul de la divergence et normalisation
        divergence = divergence_coeff / diffusivity_map
        culture_L2 = xp.sqrt(xp.sum(xp.square(culture), axis=2))
        culture_filtered = ndimage.gaussian_filter(culture, order=[0, 0, 0], sigma=sigma_culture_3d)
        
        # Vectorisation complète de dC
        divergence_term = divergence[:, :, xp.newaxis] / (divergence + culture_L2)[:, :, xp.newaxis]
        growth_culture = divergence_term * culture_filtered * (1 - culture_L2[:, :, xp.newaxis] / culture_limit)
        dC = xp.where(population[:, :, xp.newaxis] > 0, culture_diffusion, 0) + growth_culture
        dC = xp.nan_to_num(dC, nan=0)  # Gestion des NaN
        
        # Mise à jour de la culture avec normalisation
        culture += dC * time_step
        culture_L2_updated = xp.sqrt(xp.sum(xp.square(culture), axis=2))
        scale_factor = xp.where(culture_L2_updated > culture_limit, 
                            culture_limit / culture_L2_updated, 
                            1.0)[:, :, cp.newaxis]
        culture *= scale_factor

        # Mettre à jour les attributs
        self.population = population
        self.culture = culture
        self.population_diffusion = population_diffusion
        self.culture_diffusion = culture_diffusion
if __name__ == "__main__":

    video = False

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

    if not video:
        obj.view_fields()

    concurrent.futures.wait(obj.display_tasks)

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
    command = ['ffmpeg','-hwaccel', 'cuda','-f','concat','-r','25','-y','-i',image_list_filename,video_name]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stderr.decode())