# coding: utf-8
import time
from flask import Flask, request
from flask_restplus import Resource, Api
from flask_cors import CORS
from flask_restful_swagger import swagger
from comparison import compare_images


app_name = 'Face comparison'
app = Flask(__name__)
api = swagger.docs(Api(version='1.0', 
                       title=app_name, 
                       description='Face comparison API', 
                       default_label='Tasks',
                       default='/api/v1'))

def configure_app(app, config=None):
    CORS(app, resources={r'*': {'origins': '*'}})
    api.init_app(app)

@api.route('/detector')
@api.doc(params={'image1': 'Image 1 url'})
@api.doc(params={'image2': 'Image 2 url'})
@api.doc(params={'threshold': 'Threshold'})
@api.doc(params={'detection_type': 'Strong detection?'})
class Comparator(Resource):
    def get(self):
        print("Processing urls: [{}, {}]".format(request.args['image1'], request.args['image2']))
        start = time.time()
        image1 = request.args['image1']
        image2 = request.args['image2']
        threshold = request.args.get('threhsold', 0.99)
        detection_type = request.args.get('detection_type')
        print("Detection type {}".format(detection_type))
        is_similar = compare_images(image1, image2, threshold, detection_type)
        print("Execution took {:.4f}".format(time.time() - start))
        return is_similar

@api.route('/ping')
class Health(Resource):
    def get(self):
        return "pong"

if __name__ == '__main__':
    configure_app(app, {'PROJECT': app_name + '-service'})
    app.run(debug=True,host='0.0.0.0')
