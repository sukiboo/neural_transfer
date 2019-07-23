

import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import vgg19


''' input data '''
# images
content_image_name = 'turtle.jpg'
style_image_name = 'kandinsky.jpg'

# layers and weights
content_layer_index = [20]
content_layer_weight = [1]
style_layer_index = [4, 7, 12]#[1, 4, 7, 12, 17]#[6, 11, 16]
style_layer_weight = [1, 3, 1]#[1, 10, 100, 1000, 100]

# hyperparameters
iterations = 1000
content_weight = 1e+00
style_weight = 1e+02
variation_weight = 1e+00
learning_rate = 1e-02

# settings
display_images = True
display_layers = True
display_model = False



''' nu cho narod pognali ebaniy nahui '''
# enable eager execution
tf.enable_eager_execution()
# suppress warnings
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



''' load images '''
# load and resize an image
def load_and_resize_image(path_to_image, max_dim=512):
	# load image
	image = tf.io.decode_image(tf.io.read_file(path_to_image), dtype=tf.float32)
	# resize image
	image_size = tf.cast(tf.shape(image)[:-1], dtype=tf.float32)
	scale = tf.cast(max_dim / max(image_size), dtype=tf.float32)
	image = tf.image.resize(image, tf.cast(image_size*scale, dtype=tf.int32))
	return image

# load and resize content and style images
content_image = load_and_resize_image('./input_images/' + content_image_name)
style_image = load_and_resize_image('./input_images/' + style_image_name)
# initialize generated image as the content image
transfer_image = tf.Variable(content_image)



''' define the model '''
# load pre-trained feature extractor
cnn = vgg19.VGG19(\
		include_top = False,\
		weights = 'imagenet',\
		)#input_shape = cnn_input_size)

# get content and style layers
content_layers = [cnn.layers[k] for k in content_layer_index]
style_layers = [cnn.layers[k] for k in style_layer_index]
# display layers
if display_layers:
	print('\ncnn layers:', *map(lambda layer: '{:2d} -- {:s}'\
		.format(cnn.layers.index(layer), layer.name), cnn.layers), sep='\n  ')
	print('\ncontent layers:', *map(lambda layer: '{:2d} -- {:s}'\
		.format(cnn.layers.index(layer), layer.name), content_layers), sep='\n  ')
	print('\nstyle layers:', *map(lambda layer: '{:2d} -- {:s}'\
		.format(cnn.layers.index(layer), layer.name), style_layers), sep='\n  ')
	print()

# construct the model that outputs activations of content and style layers
model = Model(inputs=[cnn.input],\
			outputs=dict([(layer.name, layer.output)\
			for layer in content_layers + style_layers]))
model.trainable = False
# display model summary
if display_model:
	model.summary()

# extract features of a given image
def extract_features(image):
	# preprocess the image for the cnn
	cnn_image = vgg19.preprocess_input(tf.expand_dims(image*255, axis=0))
	# propagate image through the cnn
	outputs = model(cnn_image)
	return outputs



''' define content loss function '''
# extract content image features
content_outputs = extract_features(content_image)
content_features = dict([\
					(layer.name, content_outputs[layer.name])\
					for layer in content_layers])

# compute content loss
def compute_content_loss(outputs):
	# compute square loss for each content layer
	content_layer_loss = [tf.reduce_sum((outputs[layer.name] - content_features[layer.name])**2)\
							/ tf.reduce_sum(content_features[layer.name]**2)\
							for layer in content_layers]
	# weight and sum content layer losses according to content layer weights
	content_loss = tf.reduce_sum(tf.multiply(content_layer_loss, content_layer_weight))\
				/ tf.cast(tf.reduce_sum(tf.abs(content_layer_weight)), dtype=tf.float32)
	return content_loss



''' define style loss function '''
# compute gram matrix of a 4-array of the shape (batch, I, J, channels)
def compute_gram_matrix(feature_array):
	gram_matrix = tf.linalg.einsum('bijc,bijd->bcd', feature_array, feature_array)\
		/ tf.cast(tf.reduce_prod(feature_array.shape[1:-1]), tf.float32)
	return gram_matrix

# extract the features of content and style images and compute their gram matrix
style_outputs = extract_features(style_image)
style_gram_matrix = dict([(layer.name, compute_gram_matrix(style_outputs[layer.name]))\
						for layer in style_layers])
content_gram_matrix = dict([(layer.name, compute_gram_matrix(content_outputs[layer.name]))\
						for layer in style_layers])

# compute style loss
def compute_style_loss(outputs):
	# compute square loss for each style layer
	style_layer_loss = [\
		tf.reduce_sum((compute_gram_matrix(outputs[layer.name]) - style_gram_matrix[layer.name])**2)\
		/ tf.reduce_sum((content_gram_matrix[layer.name] - style_gram_matrix[layer.name])**2)\
		for layer in style_layers]
	# weight and sum style layer losses according to style layer weights
	style_loss =  tf.reduce_sum(tf.multiply(style_layer_loss, style_layer_weight))\
				/ tf.cast(tf.reduce_sum(tf.abs(style_layer_weight)), dtype=tf.float32)
	return style_loss



''' define variation loss function '''
# compute variation of content and style images
horizontal_variation_content = content_image[:,:-1,:] - content_image[:,1:,:]
vertical_variation_content = content_image[:-1,:,:] - content_image[1:,:,:]
horizontal_variation_style = style_image[:,:-1,:] - style_image[:,1:,:]
vertical_variation_style = style_image[:-1,:,:] - style_image[1:,:,:]

# compute total variation loss
def compute_variation_loss(image):
	horizontal_variation = image[:,:-1,:] - image[:,1:,:]
	vertical_variation = image[:-1,:,:] - image[1:,:,:]
	variation_loss = tf.reduce_mean((horizontal_variation - horizontal_variation_content)**2)\
						+ tf.reduce_mean((vertical_variation - vertical_variation_content)**2)\
						/ (tf.reduce_mean(horizontal_variation_content**2)\
							+ tf.reduce_mean(vertical_variation_content**2)\
							+ tf.reduce_mean(horizontal_variation_style**2)\
							+ tf.reduce_mean(vertical_variation_style**2))
	return variation_loss



''' perform neural style transfer '''
# select optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# perform a step of optimization algorithm
@tf.function
def optimize_image(image):

	with tf.GradientTape() as tape:
		# propagate image through the network
		outputs = extract_features(image)
		# compute losses
		content_loss = content_weight * compute_content_loss(outputs)
		style_loss = style_weight * compute_style_loss(outputs)
		variation_loss = variation_weight * compute_variation_loss(image)
		total_loss = content_loss + style_loss + variation_loss

	# compute gradients
	gradients = tape.gradient(total_loss, image)
	# optimize an image
	optimizer.apply_gradients([(gradients, image)])
	image.assign(tf.clip_by_value(image, 0, 1))

	return content_loss, style_loss, variation_loss


# save an image
def save_image(image, n=0, image_format='png'):
	# #name = '_'.join([content_image_name.split('.')[0], style_image_name.split('.')[0], str(n)])
	# #name = str(style_layer_index) + 'x' + str(style_layer_weight) + '_' + str(n)
	name = '_'.join([content_image_name.split('.')[0],\
		style_image_name.split('.')[0], str(style_layer_index), str(style_layer_weight), str(n)])
	tf.keras.preprocessing.image.save_img(\
		'./output_images/' + name + '.' + image_format,\
		image,\
		file_format = image_format)
	return


# generate an image
print('\nperforming neural style transfer...')
save_image(transfer_image.read_value(), 0)
for i in range(iterations):

	# report individual losses
	if (i + 1) % 1 == 0:
		outputs = extract_features(transfer_image)
		cnt = [tf.reduce_sum((outputs[layer.name] - content_features[layer.name])**2)\
			/ tf.reduce_sum(content_features[layer.name]**2)\
			for layer in content_layers]
		# #cnt = tf.multiply(cnt, content_layer_weight)
		stl = [tf.reduce_sum((compute_gram_matrix(outputs[layer.name]) \
			- style_gram_matrix[layer.name])**2)\
			/ tf.reduce_sum((content_gram_matrix[layer.name] - style_gram_matrix[layer.name])**2)\
			for layer in style_layers]
		# #stl = tf.multiply(stl, style_layer_weight)
		print('content layer losses:', *map(lambda x: '{:.2e}'.format(x.numpy()), cnt), sep='\n  ')
		print('style layer losses:', *map(lambda x: '{:.2e}'.format(x.numpy()), stl), sep='\n  ')
		tvr = compute_variation_loss(transfer_image)
		print('total variation losses:', '{:.2e}'.format(tvr), sep='\n  ', end='\n\n')

	# perform a step of neural transfer
	content_loss, style_loss, variation_loss = optimize_image(transfer_image)
	if (i + 1) % 100 == 0:
		save_image(transfer_image.read_value(), i+1)
		print('  {:3d} / {:d} -- content/style/variation loss: {:.2e} / {:.2e} / {:.2e}'\
			.format(i+1, iterations, content_loss, style_loss, variation_loss))


# display content, transfer, and style images
if display_images:
	# create figure
	plt.figure(figsize=(20,6))
	# content image
	plt.subplot(1,3,1)
	plt.title('content image')
	plt.imshow(content_image)
	# transfer image
	plt.subplot(1,3,2)
	plt.title('transfer image')
	plt.imshow(transfer_image.read_value())
	# style image
	plt.subplot(1,3,3)
	plt.title('style image')
	plt.imshow(style_image)
	# show images
	plt.tight_layout()
	plt.show()

