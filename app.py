# coding: utf-8
import time
import cStringIO as StringIO
import optparse
import tornado.wsgi
import tornado.httpserver
import logging

from flask import Flask, request
from flask_restplus import Resource, Api
from flask_cors import CORS
from flask_restful_swagger import swagger
from comparison import face_detect, compare_images
from PIL import Image
from classifier import ImageClassifier

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


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


@api.route('/detector')
@api.doc(params={'image1': 'Image 1 url'})
@api.doc(params={'image2': 'Image 2 url'})
@api.doc(params={'threshold': 'Threshold'})
@api.doc(params={'detection_type': 'Strong detection?'})
class Comparator(Resource):
    def get(self):
        print("Processing urls: [{}, {}]".format(request.args['image1'],
                                                 request.args['image2']))
        start = time.time()
        url1 = request.args['image1']
        url2 = request.args['image2']
        threshold = request.args.get('threhsold', 0.99)
        detection_type = request.args.get('detection_type')
        strong_detection = 1 if detection_type else -1
        print("Performing strong detection? {}".format(strong_detection == -1))

        # Face detection
        box1, image1 = face_detect(url1, strong_detection)
        box2, image2 = face_detect(url2, strong_detection)
        print("Detection took {:.4f}".format(time.time() - start))
        # Document classification
        class_start = time.time()
        is_document_A = app.document_classifier.classify_image(image1)
        is_document_B = app.document_classifier.classify_image(image2)
        print("Classification took {:.4f}".format(time.time() - class_start))

        # image comparison
        comp_start = time.time()
        is_similar = compare_images(image1, box1, image2, box2,
                                    threshold, detection_type)
        print("Face comparison took {:.4f}".format(time.time() - comp_start))

        print("Execution took {:.4f}".format(time.time() - start))
        return {'imageA': {'url': url1, 'is_document': is_document_A},
                'imageB': {'url': url2, 'is_document': is_document_B},
                'is_similar': is_similar}


@api.route('/ping')
class Health(Resource):
    def get(self):
        return "pong"


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    configure_app(app, {'PROJECT': app_name + '-service'})
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImageClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.document_classifier = ImageClassifier(**ImageClassifier.default_args)
    app.document_classifier.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
