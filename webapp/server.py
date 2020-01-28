from flask import Flask, render_template, request
import imageio
from pathlib import Path
import os.path
import shutil
import random

from fastai.vision import *


# Create the application object
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


learn = load_learner(r"C:\Users\William\OneDrive\Code\Output","First Attempt.pkl")

demo_groups = {'Demo_1':[r"C:\Users\William\OneDrive\Code\Output\_C1yuZv7hC.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_Bz9k-jx2k.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_A93IzwCPx.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_BEZenHvU9.jpg"              
						],
				'Demo_2':[r"C:\Users\William\OneDrive\Code\Output\_9r0DbP7oj.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_3BbeqQCIA.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_cS3WsqtAp.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_ggkdaDxz6.jpg"],
				
				'Demo_3':[r"C:\Users\William\OneDrive\Code\Output\_9r0DbP7oj.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_3BbeqQCIA.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_cS3WsqtAp.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_ggkdaDxz6.jpg"],

				'Demo_4':[r"C:\Users\William\OneDrive\Code\Output\_9r0DbP7oj.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_3BbeqQCIA.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_cS3WsqtAp.jpg",
						r"C:\Users\William\OneDrive\Code\Output\_ggkdaDxz6.jpg"]
				}

cache = {}



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
		some_output="The predicted engagement for that post is:"

		some_image, image_exists = open_image_path(some_input)
		some_number = get_pred(some_input, learn) if image_exists else 0.0
		output_message = 'Image loaded successfully' if image_exists else 'Cannot open file'
		return render_template("index.html",
							output_message = output_message,
                          	my_input="",
                          	my_output=some_output,
                          	my_number=some_number,
                          	my_img_name=some_image,
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

	predictions = get_predictions(processed_images)

	best_image = get_best_image(processed_images, predictions)

	result_path, alt_path = make_composite_image(processed_images, predictions)

	cache['best_image'] = best_image
	cache['results_image'] = result_path
	return render_template("index.html",
					output_message = output_message,
					#my_input="",
					#my_output=some_output,
					#my_number=some_number,
					best_image_name = best_image,
					my_img_name=result_path,
					my_form_result="NotEmpty",
					optimize_pane = 'Unoptimized')



@app.route('/optimize')
def optimize_post():
	return render_template("index.html",
			output_message = "",
			#my_input="",
			#my_output=some_output,
			#my_number=some_number,
			best_image_name = cache['best_image'],
			my_img_name=cache['results_image'],
			my_form_result="NotEmpty",
			post_date = 'Wednesday at 1:00',
			filter_name = 'None',
			optimize_pane = 'Optimized')


def get_best_image(img_list, predictions, base_path = Path(r'static/temp')):
	max_indices = {index.item() for index in np.where(predictions == np.max(predictions))[0]}
	best_index = list(max_indices)[0]
	best_image = img_list[best_index]
	best_image.save(base_path/'best_image.png')
	return 'best_image.png'
	
def get_pred(img, learn, decimals = 2):
	return round(random.random(), decimals)
	img_tens = open_image(img)
	result = (learn.predict(img_tens)[0].obj).item()
	return round(result, decimals)

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

		return filename, True	#return inp_path
		#return imageio.imread(inp_path)	
	else: return 'invalid2.jpg', False
# start the server with the 'run()' method

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


if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/

