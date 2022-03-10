"""
This script performs a Neural Style Transfer.
For an interactive version refer to the file neural_style_transfer_colab.ipynb
"""

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19
import matplotlib.pyplot as plt


def load_and_resize_image(path_to_image, max_dim=512):
    '''load and resize an image'''
    # load image
    image = tf.io.decode_image(tf.io.read_file(path_to_image), dtype=tf.float32)
    # resize image if it is larger than max_dim
    image_size = tf.cast(tf.shape(image)[:-1], dtype=tf.float32)
    if max(image_size) > max_dim:
        scale = tf.cast(max_dim / max(image_size), dtype=tf.float32)
        image = tf.image.resize(image, tf.cast(image_size*scale, dtype=tf.int32))
    return image

def extract_features(image):
    '''extract features of a given image'''
    # preprocess the image for the cnn
    cnn_image = vgg19.preprocess_input(tf.expand_dims(image*255, axis=0))
    # propagate image through the cnn
    outputs = model(cnn_image)
    return outputs

def compute_content_loss(outputs):
    '''compute content loss'''
    # compute square loss for each content layer
    content_layer_loss = [\
        tf.reduce_sum((outputs[layer.name] - content_features[layer.name])**2) / content_scale[layer.name]\
        for layer in content_layers]
    # weight and sum content layer losses according to content layer weights
    content_loss = tf.reduce_sum(tf.multiply(content_layer_loss, content_layer_weight))\
                    / tf.cast(tf.reduce_sum(tf.abs(content_layer_weight)), dtype=tf.float32)
    return content_loss

def compute_gram_matrix(feature_array):
    '''compute gram matrix of a 4-array of the shape (batch, I, J, channels)'''
    gram_matrix = tf.linalg.einsum('bijc,bijd->bcd', feature_array, feature_array)\
                  / tf.cast(tf.reduce_prod(feature_array.shape[1:-1]), tf.float32)
    return gram_matrix

def compute_style_loss(outputs):
    '''compute style loss'''
    # compute square loss for each style layer
    style_layer_loss = [\
        tf.reduce_sum(tf.abs(compute_gram_matrix(outputs[layer.name]) - style_gram_matrix[layer.name]))\
                      / style_scale[layer.name] for layer in style_layers]
    # weight and sum style layer losses according to style layer weights
    style_loss =  tf.reduce_sum(tf.multiply(style_layer_loss, style_layer_weight))\
                  / tf.cast(tf.reduce_sum(tf.abs(style_layer_weight)), dtype=tf.float32)
    return style_loss

def compute_variation_loss(image, variation_l1_l2_weights=[0,1]):
    '''compute variation loss'''
    # compute image variation
    variation_horizontal = image[:,:-1,:] - image[:,1:,:]
    variation_vertical = image[:-1,:,:] - image[1:,:,:]
    # compute 1-norm between transfer and content images
    variation_loss_l1_horizontal = tf.reduce_sum(tf.abs(variation_horizontal - content_variation_horizontal))\
                                    * tf.reduce_sum(tf.abs(variation_horizontal))
    variation_loss_l1_vertical = tf.reduce_sum(tf.abs(variation_vertical - content_variation_vertical))\
                                    * tf.reduce_sum(tf.abs(variation_vertical))
    # scale 1-variation loss
    variation_loss_l1_horizontal *= style_variation_l1_horizontal / (content_variation_l1_horizontal + 1e-32)
    variation_loss_l1_vertical *= style_variation_l1_vertical / (content_variation_l1_vertical + 1e-32)
    variation_loss_l1 = variation_loss_l1_horizontal + variation_loss_l1_vertical
    # compute 2-norm between transfer and content images
    variation_loss_l2_horizontal = tf.reduce_sum((variation_horizontal - content_variation_horizontal)**2)\
                                    * tf.reduce_sum(variation_horizontal**2)
    variation_loss_l2_vertical = tf.reduce_sum((variation_vertical - content_variation_vertical)**2)\
                                    * tf.reduce_sum(variation_vertical**2)
    # scale 2-variation loss
    variation_loss_l2_horizontal *= style_variation_l2_horizontal / (content_variation_l2_horizontal + 1e-32)
    variation_loss_l2_vertical *= style_variation_l2_vertical / (content_variation_l2_vertical + 1e-32)
    variation_loss_l2 = tf.sqrt(1e-32 + variation_loss_l2_horizontal + variation_loss_l2_vertical)
    # weight and sum 1- and 2-variation losses
    variation_loss = tf.reduce_sum(tf.multiply([variation_loss_l1, variation_loss_l2], variation_l1_l2_weights))\
                    / tf.cast(tf.reduce_sum(tf.abs(variation_l1_l2_weights)), dtype=tf.float32)
    return variation_loss

def optimize_image(image):
    '''perform a step of optimization algorithm'''
    with tf.GradientTape() as tape:
        # propagate image through the network
        outputs = extract_features(image)
        # compute losses
        content_loss = content_weight * compute_content_loss(outputs)
        style_loss = style_weight * compute_style_loss(outputs)
        # optionally compute variation loss
        if variation_weight > 0:
            variation_loss = variation_weight * compute_variation_loss(image)
        else:
            variation_loss = .0
        # total loss
        total_loss = content_loss + style_loss + variation_loss
    # compute gradients
    gradients = tape.gradient(total_loss, image)
    # optimize an image
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, 0, 1))
    return content_loss, style_loss, variation_loss


if __name__ == '__main__':

    '''input images'''
    # select images
    content_image_name = 'kitty.jpg'
    style_image_name = 'monet.jpg'

    # load and resize content and style images
    content_image = load_and_resize_image('./content_images/' + content_image_name)
    style_image = load_and_resize_image('./style_images/' + style_image_name)
    # initialize generated image as the content image
    transfer_image = tf.Variable(content_image)


    '''model construction'''
    # define the feature extractor
    cnn = vgg19.VGG19(include_top=False, weights='imagenet')
    # display the layers
    print('\ncnn layers:', *map(lambda layer: f'{cnn.layers.index(layer):2d} -- {layer.name}', cnn.layers), sep='\n  ')

    # indices of content and style layers
    content_layer_index = [21]
    style_layer_index = [1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
    # weights of individual layers
    content_layer_weight = [1]
    style_layer_weight = [3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]

    # get content and style layers
    content_layers = [cnn.layers[k] for k in content_layer_index]
    style_layers = [cnn.layers[k] for k in style_layer_index]
    # display content and style layers
    print('\nstyle layers:',
          *map(lambda layer: f'{cnn.layers.index(layer):3d} -- {layer.name}', style_layers), sep='\n  ')
    print('\ncontent layers:',
          *map(lambda layer: f'{cnn.layers.index(layer):3d} -- {layer.name}', content_layers), sep='\n  ')

    # construct the model that outputs activations of content and style layers
    model = Model(inputs=[cnn.input],
                  outputs=dict([(layer.name, layer.output) for layer in content_layers+style_layers]))
    # freeze the model
    model.trainable = False
    # display model summary
    model.summary()


    '''loss function'''
    # extract features from the content image
    content_outputs = extract_features(content_image)
    content_features = dict([(layer.name, content_outputs[layer.name]) for layer in content_layers])
    content_scale = dict([(layer.name, tf.reduce_sum(content_features[layer.name]**2)) for layer in content_layers])

    # extract the features of content and style images and compute their gram matrix
    style_outputs = extract_features(style_image)
    style_gram_matrix = dict([(layer.name, compute_gram_matrix(style_outputs[layer.name])) for layer in style_layers])
    content_gram_matrix = dict([(layer.name, compute_gram_matrix(content_outputs[layer.name])) for layer in style_layers])
    style_scale = dict([(layer.name, tf.reduce_sum(tf.abs(content_gram_matrix[layer.name]\
                         - style_gram_matrix[layer.name]))) for layer in style_layers])

    # compute variation of the content image
    content_variation_horizontal = content_image[:,:-1,:] - content_image[:,1:,:]
    content_variation_vertical = content_image[:-1,:,:] - content_image[1:,:,:]
    # compute 1-variation of the content image
    content_variation_l1_horizontal = tf.reduce_sum(tf.abs(content_variation_horizontal))
    content_variation_l1_vertical = tf.reduce_sum(tf.abs(content_variation_vertical))
    # compute 2-variation of the content image
    content_variation_l2_horizontal = tf.reduce_sum(content_variation_horizontal**2)
    content_variation_l2_vertical = tf.reduce_sum(content_variation_vertical**2)

    # compute variation of the original style image
    style_original = tf.io.decode_image(tf.io.read_file('./style_images/' + style_image_name), dtype=tf.float32)
    style_variation_horizontal = style_original[:,:-1,:] - style_original[:,1:,:]
    style_variation_vertical = style_original[:-1,:,:] - style_original[1:,:,:]
    # compute 1-variation of the style image
    style_variation_l1_horizontal = tf.reduce_sum(tf.abs(style_variation_horizontal))
    style_variation_l1_vertical = tf.reduce_sum(tf.abs(style_variation_vertical))
    # compute 2-variation of the style image
    style_variation_l2_horizontal = tf.reduce_sum(style_variation_horizontal**2)
    style_variation_l2_vertical = tf.reduce_sum(style_variation_vertical**2)


    '''optimization settings'''
    # neural transfer parameters
    content_weight = 1
    style_weight = 100
    variation_weight = 1e-05
    # optimization parameters
    iterations = 100
    learning_rate = 1e-01
    decay_rate = 5e-02
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate, decay=decay_rate)

    # iteratively update the image
    for i in range(iterations):
        # perform a step of neural transfer
        content_loss, style_loss, variation_loss = optimize_image(transfer_image)
        # update output
        if (i + 1) % 10 == 0:
            print('  {:>4d} / {:d} -- content / style / variation loss: {:.2e} / {:.2e} / {:.2e}'\
                .format(i+1, iterations, content_loss, style_loss, variation_loss))
            tf.keras.preprocessing.image.save_img(f"./{content_image_name.split('.')[0]}_" +\
                                                  f"{style_image_name.split('.')[0]}_{style_weight}_{i+1}.png",
                                                  transfer_image.read_value(), file_format='png')
    # save the final transfer image
    tf.keras.preprocessing.image.save_img(f"./transfer_images/{content_image_name.split('.')[0]}_" +\
                                          f"{style_image_name.split('.')[0]}_{style_weight}.png",
                                          transfer_image.read_value(), file_format='png')


    '''display content, transfer, and style images'''
    plt.figure(figsize=(19,6))
    plt.subplot(1,3,1).axis('off')
    plt.title('content image')
    plt.imshow(content_image)
    plt.subplot(1,3,2).axis('off')
    plt.title('transfer image')
    plt.imshow(transfer_image.read_value())
    plt.subplot(1,3,3).axis('off')
    plt.title('style image')
    plt.imshow(style_image)
    plt.tight_layout()
    plt.show()
