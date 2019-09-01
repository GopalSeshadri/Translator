import pickle
from keras.models import model_from_json

class Utilities:
    def saveModel(model, model_name):
        model_json = model.to_json()
        with open('Models/{}.json'.format(model_name), 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('Models/{}.h5'.format(model_name))

    def saveDict(dict_obj, dict_name):
        with open('Models/{}.pickle'.format(dict_name), 'wb') as f:
            pickle.dump(dict_obj, f, protocol = pickle.HIGHEST_PROTOCOL)


    def loadModel(model_name):
        json_file = open('Models/{}.json'.format(model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Models/{}.h5".format(model_name))
        return loaded_model


    def loadDict(dict_name):
        with open('Models/{}.pickle'.format(dict_name), 'rb') as file:
            dict_obj = pickle.load(file)
        return dict_obj
