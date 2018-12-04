import coremltools

#this is a scipt to convert a caffe model in to a ML file for swift

caffe_model = ('oxford102.caffemodel', 'deploy.prototxt')

labels = 'flower-labels.txt'

coreml_model = coremltools.converters.caffe.convert(
    caffe_model,
    class_labels=labels,
    image_input_names='data'
)

coreml_model.save('FlowerClassifier.mlmodel')
