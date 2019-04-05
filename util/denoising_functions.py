import importlib
import numpy
import math
import random
from deap import algorithms, base, creator, tools
from skimage import data, io, filters, restoration
from matplotlib import pyplot as plt
from util import performance_functions
from util.image_dataset import ImageDataset
importlib.reload(performance_functions) 

def denoise_image(image, action_id):
    
    psf = numpy.ones((5, 5, 3)) / 25
    denoising_filters = [
        lambda x : (filters.gaussian(x ,sigma=0.05)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=0.1)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=0.5)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=1)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=1.5)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=2)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x, sigma=3)*255).astype(numpy.uint8),
        lambda x : (filters.gaussian(x ,sigma=4)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.06)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.05)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.04)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.03)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.02)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.01)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.005)*255).astype(numpy.uint8),
        lambda x : (restoration.denoise_tv_chambolle(x, weight=0.001)*255).astype(numpy.uint8),
    ]
    if action_id > len(denoising_filters):
        return image
    return denoising_filters[action_id](image)

def bits_to_int(bit_list):
    return sum([(x*(2**i)) for i, x in enumerate(bit_list[::-1])])

def extract_textures(image, weak_texture_mask):
    weak_texture = (image * weak_texture_mask).astype(numpy.uint8)
    strong_texture = image - weak_texture
    return strong_texture.astype(numpy.uint8), weak_texture.astype(numpy.uint8)

def excecute_actions(individual, images, image_index = 0):
    weak_texture_action = bits_to_int(individual[0:len(individual)//2])
    strong_texture_action = bits_to_int(individual[len(individual)//2:len(individual)])
    
    
    denoised_weak_texture = denoise_image(images.noisy_images[image_index], weak_texture_action)
    denoised_strong_texture = denoise_image(images.noisy_images[image_index], strong_texture_action)
    
    strong_texture, _ = extract_textures(denoised_strong_texture, images.weak_texture_masks[image_index])
    _, weak_texture = extract_textures(denoised_weak_texture, images.weak_texture_masks[image_index])
    
    return (strong_texture + weak_texture ) 

def evaluate(individual, images ,display=False, image_index = 0, performance = performance_functions.root_mean_squared_error): 
    
    denoised_image = excecute_actions(individual, images, image_index)
    
    if display:
        merged_images = numpy.hstack((images.base_images[image_index],\
                                      images.noisy_images[image_index]*255,\
                                      denoised_image))\
                             .astype(numpy.uint8)
        plt.title("Original Image(Left) | Noisy Image(Center) | Denoised Image(Right)")
        plt.imshow(merged_images)
        plt.show()
        
                                                   
                                                   
    return performance(denoised_image, images.base_images[image_index]),

def get_denoised_image(individual, images , image_index):
    return excecute_actions(individual, images, image_index)

def run_ea(noise_level = 0.005, pop = 20, generations = 10, evaluation_method = "RMSE", num_other_images = 4, display = False):
    
    images = ImageDataset(num_other_images ,noise_level)
    NUM_FILTERS = 16
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
    print(evaluate(tools.selBest(pop, k=1)[0], images, display))
    return tools.selBest(pop, k=1)[0], images, noise_level

def compare_results(individual, images, noise_level, display = False):
    for i in range(0,len(images.base_images)):  
        if i == 0:
            print("Evaluation of image ea was run on")
        else:
            print("Evaluation of unseen image %d:\n"%(i))
        new_denoised_image = get_denoised_image(individual, images, i)
        new_denoised_image_weak_filter = get_denoised_image(\
                                            individual[:len(individual)//2] + individual[:len(individual)//2],
                                            images, i)
        new_denoised_image_rich_filter = get_denoised_image(\
                                            individual[len(individual)//2:] + individual[len(individual)//2:],
                                            images, i)
        denoise_image_gaussian_blur = (filters.gaussian(images.noisy_images[i], sigma= images.noise_levels[i][0]) * 255).astype(numpy.uint8)

        if display:
            image_and_wtm = numpy.hstack((images.base_images[i], images.weak_texture_masks[i]*255)).astype(numpy.uint8)
            noisy_and_denoised_image = numpy.hstack((images.noisy_images[i]*255, new_denoised_image)).astype(numpy.uint8)
            image_stages_merged = numpy.vstack((image_and_wtm,noisy_and_denoised_image))
            plt.rcParams["figure.figsize"]=20,20
            plt.title("Original Image(Left) | Weak Texture Mask | Denoised Image")
            plt.imshow(image_stages_merged)
            plt.show()
        
        print("Baseline Statistics:", end=" ")
        print_statistics(images.base_images[i], (images.noisy_images[i]*255).astype(numpy.uint8))
        print("Our Method Statistics:", end=" ")
        print_statistics(images.base_images[i], new_denoised_image)
        print("Weak Texture Filter Only Statistics:", end=" ")
        print_statistics(images.base_images[i], new_denoised_image_weak_filter)
        print("Rich Texture Filter Only Statistics:", end=" ")
        print_statistics(images.base_images[i], new_denoised_image_rich_filter)
        print("Standard Gaussian Blur Statistics:", end=" ")
        print_statistics(images.base_images[i], denoise_image_gaussian_blur)
        
        
def print_statistics(image, evaluation_image):
        print("RMSE: %f | PSNR: %f | IQI: %f | SSIM: %f" % \
             (performance_functions.root_mean_squared_error(image, evaluation_image),\
              performance_functions.peak_signal_noise_ratio(image, evaluation_image),\
              performance_functions.image_quality_index(image, evaluation_image),\
              performance_functions.structural_similarity_indix(image, evaluation_image),))
    