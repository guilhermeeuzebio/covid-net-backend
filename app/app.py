from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os
import json

app = Flask(__name__)

weightspath = 'models/COVIDNet-CXR-Large'
metaname = 'model.meta'
ckptname = 'model-8485'

@app.route('/api/v1/covid-inference', methods=['POST'])
def covid_inference():
	try:
		file = request.files['file']
		filename = file.filename
		file.save(os.path.join('images', filename))
		
		mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
		inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

		sess = tf.Session()
		tf.get_default_graph()
		saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))
		saver.restore(sess, os.path.join(weightspath, ckptname))

		graph = tf.get_default_graph()

		image_tensor = graph.get_tensor_by_name("input_1:0")
		pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

		x = cv2.imread('images/'+filename)
		h, w, c = x.shape
		x = x[int(h/6):, :]
		x = cv2.resize(x, (224, 224))
		x = x.astype('float32') / 255.0
		pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

		return jsonify({'prediction': inv_mapping[pred.argmax(axis=1)[0]], 'normal': str(pred[0][0]), 'pneumonia': str(pred[0][1]), 'covid_19': str(pred[0][2]), 'status': 200}), 200
	except Exception as e:
		print('ERROR: {}'.format(e))
		return jsonify({'error': 'server did not recive data to process', 'status': 400}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
