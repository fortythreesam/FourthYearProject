import importlib
import numpy
import math
import random
from deap import algorithms, base, creator, tools
from skimage import data, io, filters, restoration
from matplotlib import pyplot as plt
from util import performance_functions
importlib.reload(performance_functions) 


def denoise_image(image, action_id):
    
    psf = numpy.ones((5, 5, 3)) / 25
    actions = [
        lambda x : (filters.gaussian(x ,sigma=0.01)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=0.02)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=0.025)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=0.4)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.02)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.01)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.005)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.001)*255).astype(numpy.uint8),
    ]
    if action_id > len(actions):
        return image
    return actions[action_id](image)

def bits_to_int(bit_list):
    return sum([(x*(2**i)) for i, x in enumerate(bit_list[::-1])])

def extract_weak_texture(image, weak_texture_mask):
    weak_texture = (image * weak_texture_mask).astype(numpy.uint8)
    strong_texture = image - weak_texture
    return strong_texture.astype(numpy.uint8), weak_texture.astype(numpy.uint8)

def excecute_actions(individual, images, image_index = 0):
    weak_texture_action = bits_to_int(individual[0:len(individual)//2])
    strong_texture_action = bits_to_int(individual[len(individual)//2:len(individual)])
    
    
    denoised_weak_texture = denoise_image(images.noisy_images[image_index], weak_texture_action)
    denoised_strong_texture = denoise_image(images.noisy_images[image_index], strong_texture_action)
    
    strong_texture, _ = extract_weak_texture(denoised_strong_texture, images.weak_texture_masks[image_index])
    _, weak_texture = extract_weak_texture(denoised_weak_texture, images.weak_texture_masks[image_index])
    
    return (strong_texture + weak_texture ) 

def evaluate(individual, images ,display=False, image_index = 0, performance = performance_functions.root_mean_squared_error): 
    
    denoised_image = excecute_actions(individual, images, image_index)
    
    if display:
        print("PSNR: %f\n IQI: %f\n SSIM: %f\n" % \
              (performance_functions.peak_signal_noise_ratio(images.base_images[image_index], denoised_image),\
               performance_functions.image_quality_index(images.base_images[image_index], denoised_image),\
               performance_functions.structural_similarity_indix(images.base_images[image_index], denoised_image),))
        plt.rcParams["figure.figsize"]=20,20
        merged_images = numpy.hstack((images.base_images[image_index],\
                                      images.noisy_images[image_index]*255,\
                                      denoised_image))\
                             .astype(numpy.uint8)
        plt.title("Original Image(Left) | Noisy Image(Center) | Denoised Image(Right)")
        plt.imshow(merged_images)
        plt.show()
        
                                                   
                                                   
    return performance(denoised_image, images.base_images[image_index]),

def run_ea(noise_level = 0.005, pop = 20, generations = 20, evaluation_method = "RMSE"):
    
    images = ImageDataset(2,noise_level)
    NUM_FILTERS = 8
    SIZE_OF_INDIVIDUAL = math.ceil(math.log2(NUM_FILTERS**2))

    weighting = 0
    if evaluation_method == "RMSE":
        evaluation = lambda i : evaluate(i, images)
        weighting = -1.0
        individual_fitness = creator.FitnessMin
    else:
        weighting = 1.0
        individual_fitness = creator.FitnessMax
        if evaluation_method == "PSNR":
            evaluation = lambda i : evaluate(i, images, performance=performance_functions.peak_signal_noise_ratio)
        elif evaluation_method == "IQI":
            evaluation = lambda i : evaluate(i, images, performance=performance_functions.image_quality_index)
        elif evaluation_method == "SSIM":
            evaluation = lambda i : evaluate(i, images, performance=performance_functions.structural_similarity_indix)

    creator.create("FitnessMin", base.Fitness, weights=(weighting,))
    creator.create("Individual", list, fitness=individual_fitness)


    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=SIZE_OF_INDIVIDUAL)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluation)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)


    pop = toolbox.population(n=20)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=False)
    print(tools.selBest(pop, k=1)[0])
    print(evaluate(tools.selBest(pop, k=1)[0], True))