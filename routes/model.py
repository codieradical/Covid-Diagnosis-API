from flask import Response, request
import json
from keras.models import load_model
import numpy as np
from keras import backend as K
import tensorflow as tf

from vis.visualization import visualize_saliency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from matplotlib.image import imread
from PIL import Image

global model

def register_routes(app):

    predict_graph = tf.Graph()
    with predict_graph.as_default():
        predict_session = tf.Session()
        with predict_session.as_default():
            print("Loading Model...")
            model = load_model('covid_newtarget_lr3_4_2_2_1.h5')
            model._make_predict_function()
            print("Done.")

    map_graph = tf.Graph()
    with map_graph.as_default():
        map_session = tf.Session()
        with map_session.as_default():
            print("Loading Linear Activation Model...")
            saliency_model = load_model('covid_newtarget_lr3_4_2_2_1_linear.h5')
            print("Done.")

    @app.route('/v1/models/covid-19-model:predict', methods=["POST"])
    def model_predict():
        print("Predicting...")

        with predict_graph.as_default():
            with predict_session.as_default():
                probabilities = np.around(model.predict([request.json["instances"]])[0], decimals=16).tolist()
                layer = np.argmax(probabilities)

        image = np.array([request.json["instances"]][0])

        A = []
        b = []
            
        print("Creating Saliency Map...")

        with map_graph.as_default():
            with map_session.as_default():
                mapp = visualize_saliency(saliency_model, -1, layer, image[0])
                background = Image.fromarray((image[0]*255).astype('uint8'), 'RGB')
                A.append(mapp)
                b.append(np.asarray(background))
                A = np.asarray(A)
                b = np.asarray(b)
                sal_map = []
                for i in range(len(A)):
                    smoothe = gaussian_filter(A[i], sigma=4)
                    sal_map.append(smoothe)
                sal_avg = np.mean(sal_map, axis = 0)
                back_avg = np.mean(b, axis = 0)
                arr = np.zeros((224,224,3))
                for i in range(len(b)):
                    arr = arr + (b[i] / len(image))
                background = Image.fromarray(np.uint8(arr))
                plt.imshow(A[0])
                plt.imshow(sal_avg)
                plt.axis('off')
                plt.imshow(background, alpha=0.6)
                import base64
                import io 
                pic_IObytes = io.BytesIO()
                plt.savefig(pic_IObytes,  format='png')
                pic_IObytes.seek(0)
                pic_hash = base64.b64encode(pic_IObytes.read())

        saliency = pic_hash.decode("utf-8")

        return Response(json.dumps({
            "probabilities": probabilities,
            "saliency": saliency
        }))

