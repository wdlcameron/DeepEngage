from flask import Flask, render_template, request
import imageio
from pathlib import Path
import os.path
import shutil
import random
import torch

from fastai.vision import *


# Create the application object
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0




def get_best_image(img_list, predictions, base_path = Path(r'static/temp'), img_name = 'best_image.png'):
	max_indices = {index.item() for index in np.where(predictions == np.max(predictions))[0]}
	best_index = list(max_indices)[0]
	best_image = img_list[best_index]
	best_image.save(base_path/img_name)
	return best_image, img_name, best_index
	
def get_pred(img, learn, decimals = 2):
	#return round(random.random(), decimals)
	#img_tens = open_image(img)
	if learn is not None: 
		result = (learn.predict(img)[0].obj).item()
		return round(result, decimals)
	else: return round(random.random(), decimals)

def get_predictions(img_array, learn = None):
	predictions = []
	for img in img_array:
		predictions.append(get_pred(img, learn))
	return predictions

def open_image_path(inp_path):
	if os.path.exists(inp_path):
		filename = os.path.basename(inp_path)
		base_path = Path(r'static/temp')
		base_path.mkdir(exist_ok=True)
		shutil.copyfile(inp_path, base_path/filename)

		return filename	#return inp_path
		#return imageio.imread(inp_path)	
	else: return None
# start the server with the 'run()' method


def apply_filters_to_image(img, filters):
	filtered_images = []
	for f_list in filters:
		filtered_images.append(filter_image(img, f_list))
	return filtered_images

def read_and_process_images(list_of_images, size = 406):
    #size = 400  #Note: tie this variable to the model size
    base_name = 'Output_Image_'
    processed_images = [None]*len(list_of_images)
    for i, img_path in enumerate(list_of_images):
        raw_img = open_image(img_path)
        img = raw_img.apply_tfms([crop_pad()], size=400, resize_method=ResizeMethod.CROP, padding_mode='zeros')
        img.save(f'{base_name}{i}.png')
        processed_images[i] = img
    return processed_images

def make_composite_image(list_of_images, predictions = None, base_path = Path(r'static/temp')):
	main_image_name = 'predictions.png'
	alt_image_name = 'original_image.png'
	if predictions is not None: 
		assert len(list_of_images) == len(predictions), 'Number of predictions does not equal number of images (size mismatch)'
	max_indices = {index.item() for index in np.where(predictions == np.max(predictions))[0]}
    
	

	fig, axs = plt.subplots(2,2, figsize = (5,5))
	for i, ax in enumerate(axs.flatten()):    
		img, pred = list_of_images[i], predictions[i]
		img.show(ax = ax, alpha = 0.5)
		ax.axis('off')
		
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.tight_layout()
	plt.savefig(str(base_path/alt_image_name), dpi = 100, bbox_inches = 'tight',
			pad_inches = 0, transparent=True)
	
	for i, ax in enumerate(axs.flatten()):
		c, h, w = img.shape
		pred = predictions[i]
		if i not in max_indices: 
				overlay = np.ones((h,w,c), dtype='uint8')*255
				ax.imshow(overlay, alpha = 0.7)

		if predictions[i] is not None:
			ax.text(0.5, 0.5, pred, horizontalalignment='center',  
				fontsize = 24, color = 'blue', verticalalignment='center', 
				backgroundcolor = 'white', transform=ax.transAxes)

		
		plt.savefig(str(base_path/main_image_name), dpi = 100, bbox_inches = 'tight',
			pad_inches = 0, transparent=True)
	return main_image_name, alt_image_name


def mod_image(img, brightness_value=0.5, contrast_value=1.0):
    new_tfms = [brightness(change = (brightness_value, brightness_value), p = 1),
               contrast(scale = (contrast_value, contrast_value), p = 1)]
    modded_img = img.apply_tfms(new_tfms)
    return modded_img

def mod_array(img, a=0, m = 1, b = 0):
    """Mods the array in (x+a)*m + b"""
    data = img.data.clone()
    data = (data+a)*m+b
    data [data>1] = 1
    data[data<0] = 0
    return Image(data)
    
    
def filter_image(img, funcs):
    for f in funcs:   img = f(img)
    return img
    
def copy_img(img):
    data = img.data.clone()
    return Image(data)



def make_optimized_image(original_image, optimized_image, filtered_images, time_recc = "Tuesday at 4", 
						filter_recc = 'ColorPop', predictions = None, base_path = Path(r'static/temp')):
	
	image_name = 'optimized_layout.png'
	fig = plt.figure(constrained_layout = True, figsize = (4,4))
	gs = fig.add_gridspec(8, 8)
	axes = {}
	axes['ax_original'] = fig.add_subplot(gs[:4, :4])
	axes['ax_original'].set_title('Original Image')
	axes['ax_optimized'] = fig.add_subplot(gs[:4,4: ])
	axes['ax_optimized'].set_title('Optimized Image')
	axes['ax_filter_recc'] = fig.add_subplot(gs[4, :])
	axes['ax_time_recc'] = fig.add_subplot(gs[5, :])
	axes['axs_other_options'] = [fig.add_subplot(gs[6:8, 2*x:2*x+2]) for x in range(4)]

	axes['ax_filter_recc'].text(0.1, 0.25, f'Your recommended filter is {filter_recc}', multialignment='center')
	axes['ax_time_recc'].text(0.1, 0.25, f'Your recommended post time is {time_recc}', multialignment='center')


	for name, axis in axes.items():
		axs = axis if isinstance(axis, list) else [axis]
		for ax in axs:
				ax.axis('off')

				
	original_image.show(ax = axes['ax_original'])
	optimized_image.show(ax = axes['ax_optimized'])
	#ax_optimized.axis('off')

	for i, (img_tuple, ax) in enumerate(zip(filtered_images, axes['axs_other_options'])):
		name, img = img_tuple
		img.show(ax = ax)
		ax.set_title(name, fontsize = 8)
		if predictions is not None: 
			ax.text(0.5, 0.5, predictions[i], horizontalalignment='center',  
			fontsize = 8, color = 'blue', verticalalignment='center', 
			backgroundcolor = 'white', transform=ax.transAxes)
		
	plt.savefig(base_path/image_name, transparent = True, dpi =300, bbox_inches = "tight" )

	return image_name


def plot_images(img):
	base_path = Path(r'static/temp')
	
	fig, axs = plt.subplots(2,2, )
	for ax in axs.flatten():
		ax.imshow(img, alpha = 0.2)
		ax.axis('off')
		ax.text(0.5, 0.5, '0.1', horizontalalignment='center',  fontsize = 24, color = 'blue', verticalalignment='center', transform=ax.transAxes)
		#ax.patch.set_edgecolor('black')  

		#ax.patch.set_linewidth('1')  
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.tight_layout()
	plt.savefig(str(base_path/'test.png'), dpi = 100, bbox_inches = 'tight',
		pad_inches = 0)
	return 'test.png'

def channel_adjust(channel, values):
    #Adjusted from https://www.practicepython.org/blog/2016/12/20/instagram-filters-python.html
    channel_copy = channel.data.clone()
    orig_size = channel_copy.shape
    flat_channel = channel_copy.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)
    result = torch.tensor(adjusted.reshape(orig_size))
    channel[:,:] = result[:,:]
    return (channel)

def split_channels(img_array): return img_array[0,:,:],img_array[1,:,:],img_array[2,:,:]

def merge_channels(r,g,b):  return torch.tensor(np.stack([r,g,b], axis = 0))


def apply_preset_filter(img, r_values = [0, 0.5, 1], g_values = [0, 0.5, 1], b_values = [0, 0.5, 1]):
    #r,g,b = split_channels(img_array)
    data = img.data.clone()
    data[0,:,:] = channel_adjust(data[0,:,:], r_values)
    data[1,:,:] = channel_adjust(data[1,:,:], g_values)
    data[2,:,:] = channel_adjust(data[2,:,:], b_values)
    
    #merged = merge_channels(r_adjusted,g_adjusted,b_adjusted)
    data [data>1] = 1
    data[data<0] = 0
    return Image(data)


gotham_filter = partial(apply_preset_filter, 
			r_values = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
			g_values = [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 
						0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])


clareton_filter = partial(apply_preset_filter, 
    		b_values =  [x/255 for x in [0, 38, 66, 104, 139, 175, 206, 226, 245 , 255]],
    		r_values =  [x/255 for x in [0, 16, 35, 64, 117, 163, 200, 222, 237, 249]],
    		g_values =  [x/255 for x in [0, 24, 49, 98, 141, 174, 201, 223, 239, 255]])
















DEMO_PATH = Path('demo_images')

demo_groups = {'Demo_1':[DEMO_PATH/"_C1yuZv7hC.jpg",
						DEMO_PATH/"_Bz9k-jx2k.jpg",
						DEMO_PATH/"_A93IzwCPx.jpg",
						DEMO_PATH/"_BEZenHvU9.jpg"              
						],
				'Demo_2':[DEMO_PATH/"_9r0DbP7oj.jpg",
						DEMO_PATH/"_3BbeqQCIA.jpg",
						DEMO_PATH/"_cS3WsqtAp.jpg",
						DEMO_PATH/"_ggkdaDxz6.jpg"],
				
				'Demo_3':[DEMO_PATH/"_9r0DbP7oj.jpg",
						DEMO_PATH/"_3BbeqQCIA.jpg",
						DEMO_PATH/"_cS3WsqtAp.jpg",
						DEMO_PATH/"_ggkdaDxz6.jpg"],

				'Demo_4':[DEMO_PATH/"_9r0DbP7oj.jpg",
						DEMO_PATH/"_3BbeqQCIA.jpg",
						DEMO_PATH/"_cS3WsqtAp.jpg",
						DEMO_PATH/"_ggkdaDxz6.jpg"]
				}



"""The cache has key paths and images
Processed: This is true when everything has been processed, resets when you load a new group

best_image: the Image instance of the image with the highest engagement prediction
optimized_image: The Image instance of the optimized image
filtered_images: an array of the filtered images


Optimized_Display_Image: The filename for the image that summarizes the optimization figure
Prediction_Display_Image: The filename for the results of the model predictions and results
"""
cache = {'Processed': False,
		'Prediction_Display_Image':'invalid.jpg',
		'Optimized_Display_Image':'invalid.jpg',
		'Best_Display_Image':'invalid.jpg',
		'filter_names': ['ColorPop', 'Intensify', 'Gotham','Clareton'],
		'filter_sets': [[partial(mod_array, a=0,b=0,m=1), partial(mod_image, brightness_value = 0.5, contrast_value = 1.4)],
						[partial(mod_array, a=-0.1,b=0,m=1), partial(mod_image, brightness_value = 0.6, contrast_value = 1.8)],
						[gotham_filter],
						#[partial(mod_array, a=0.4,b=-0.5,m=1.1), partial(mod_image, brightness_value = 0.5, contrast_value = 2)],
						[clareton_filter]
						],

           }



cache['learn'] = load_learner("models","Second Attempt.pkl")



@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
	return render_template('index.html')  # render a template

@app.route('/output')
def recommendation_output():
#  	 
   	# Pull input
	some_input =request.args.get('user_input')

	print (some_input) 
  	 
   	# Case if empty
	if some_input == '':
		return render_template("index.html",
                              	my_input = some_input,
                              	my_form_result="Empty")
	else:
		#some_output="The predicted engagement for that post is:"
		learn = cache['learn']
		img_path = open_image_path(some_input)

		
		output_message = 'Image loaded successfully' #if image_exists else 'Cannot open file'
		
		img = read_and_process_images([some_input])[0]
		some_number = get_pred(img, learn) #if image_exists else 0.0
		cache['best_image'], cache['Best_Display_Image'] = img, img_path
		
		cache['Prediction_Display_Image'] = img_path
		return render_template("index.html",
							output_message = output_message,
                          	my_input="",
                          	#my_output=some_output,
                          	my_number=some_number,
                          	my_img_name=cache['Best_Display_Image'],
                          	my_form_result="NotEmpty",
							optimize_pane = 'Unoptimized')




@app.route('/preset')
def preset_output():
	form_data = request.args.get('demo_input')
	output_message = 'Selection' + form_data

	# list_of_images = [r"C:\Users\William\OneDrive\Code\Output\_C1yuZv7hC.jpg",
    #               r"C:\Users\William\OneDrive\Code\Output\_Bz9k-jx2k.jpg",
    #               r"C:\Users\William\OneDrive\Code\Output\_A93IzwCPx.jpg",
    #               r"C:\Users\William\OneDrive\Code\Output\_BEZenHvU9.jpg"              
    #              ]
	list_of_images = demo_groups[form_data]


	processed_images = read_and_process_images(list_of_images)

	predictions = get_predictions(processed_images, cache['learn'])

	best_image, best_image_path, best_index = get_best_image(processed_images, predictions)

	result_path, alt_path = make_composite_image(processed_images, predictions)

	cache['best_image'], cache['Best_Display_Image'] = best_image, best_image_path
	cache['Prediction_Display_Image'] = result_path
	cache['best_prediction'] = predictions[best_index]

	return optimize_post()
	


	# return render_template("index.html",
	# 				output_message = output_message,
	# 				#my_input="",
	# 				#my_output=some_output,
	# 				#my_number=some_number,
	# 				my_img_name=cache['Prediction_Display_Image'],
	# 				my_form_result="NotEmpty",
	# 				optimize_pane = 'Unoptimized')



@app.route('/optimize')
def optimize_post():
	
	def get_img(): 
		img = open_image(DEMO_PATH/"_C1yuZv7hC.jpg")
		img = img.apply_tfms([crop_pad()], size=400, resize_method=ResizeMethod.CROP, padding_mode='zeros')
		return img

	
	time_recc = "Tuesday at 4"
	filter_recc = 'ColorPop'

	original_image = cache['best_image']
	#optimized_image = get_img()
	original_prediction = cache['best_prediction']

	filter_names = cache['filter_names']
	filter_funcs = cache['filter_sets']
	filtered_images_array = apply_filters_to_image(original_image, filter_funcs)
	predictions = get_predictions(filtered_images_array, cache['learn'])
	optimized_image, best_image_path, best_index = get_best_image(filtered_images_array, predictions, img_name = 'optimized_best_image.png')
	filtered_images = [(name, img) for name, img in zip (filter_names, filtered_images_array)]



	(optimized_image, filter_recc) = (optimized_image, filter_names[best_index]) if predictions[best_index] > original_prediction else (original_image, 'Original')



	output_image_name = make_optimized_image(original_image = original_image, 
									optimized_image = optimized_image, 
									filtered_images = filtered_images, 
									time_recc = time_recc, 
									filter_recc = filter_recc, predictions = predictions)

	cache['Optimized_Display_Image'] = output_image_name


	return render_template("index.html",
			output_message = "",
			#my_input="",
			#my_output=some_output,
			#my_number=some_number,
			optimized_image_name = cache['Optimized_Display_Image'],
			my_img_name=cache['Prediction_Display_Image'],
			# my_form_result="NotEmpty",
			# post_date = 'Wednesday at 1:00',
			# filter_name = 'None',
			my_form_result="NotEmpty",
			optimize_pane = 'Optimized')



if __name__ == "__main__":
	app.run(host='0.0.0.0', debug = True) #will run locally http://127.0.0.1:5000/

