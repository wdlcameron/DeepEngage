from flask import Flask, render_template, request
import imageio
from pathlib import Path
import os.path
import shutil
import random

from fastai.vision import *


# Create the application object
app = Flask(__name__)
learn = load_learner(r"C:\Users\William\OneDrive\Code\Output","First Attempt.pkl")


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
		some_number = get_prediction(some_input, learn) if image_exists else 0.0
		output_message = 'Image loaded successfully' if image_exists else 'Cannot open file'
		return render_template("index.html",
							output_message = output_message,
                          	my_input="",
                          	my_output=some_output,
                          	my_number=some_number,
                          	my_img_name=some_image,
                          	my_form_result="NotEmpty")


def get_prediction(img, learn):
	img_tens = open_image(img)
	result = (learn.predict(img_tens)[0].obj).item()
	return result
	#return random.random()

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
if __name__ == "__main__":
	app.run(debug=True) #will run locally http://127.0.0.1:5000/

