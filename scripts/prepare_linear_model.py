# Changes output layer activation to linear for saliency maps.
# Input the model name.

import sys
from keras.models import load_model
from keras import activations
from vis.utils import utils

# eg: covid_newtarget_lr3_figure1_5
model_to_assess = sys.argv[1]

print('Preparing {} for saliency.'.format(model_to_assess))

print('Loading model...')
model = load_model('{}.h5'.format(model_to_assess))

print('Preparing model...')
# Convert output layer to linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

print('Saving model...')
model.save('{}_linear.h5'.format(model_to_assess))

print('Done!')