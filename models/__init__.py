from models.sillnet import *

def get_model(name, class_train, class_test, feature_channel):
    model = _get_model_instance(name)
        
    model = model(nc=3, input_size = 64, class_train=class_train, class_test = class_test, extract_chn=[100, 150, 200, 150, 100, feature_channel], classify_chn = [100, 150, 200, 250, 300, 100], param1 = None, param2 = None, param3 = None, param4 = [150,150,150,150])

    return model

def _get_model_instance(name):
    try:
        return {
            'sillnet' : SillNet
        }[name]
    except:
        print('Model {} not available'.format(name))