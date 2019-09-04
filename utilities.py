import pickle
from keras.models import model_from_json

class Utilities:
    def saveModel(model, model_name):
        '''
        This function takes as input a keras model and model name. Saves the provided model in Models directory.

        Parameters:
        model (keras model) : The keras model object to save.
        model_name (str) : The file name for the model to save.
        '''

        model_json = model.to_json()
        with open('Models/{}.json'.format(model_name), 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('Models/{}.h5'.format(model_name))

    def saveDict(dict_obj, dict_name):
        '''
        This function takes as input a dictionary model and dictionary name. Saves the provided dictionary in Models directory.

        Parameters:
        dict_obj (dict) : The dictionary object to save.
        dict_name (str) : The file name for the dictionary object to save.
        '''

        with open('Models/{}.pickle'.format(dict_name), 'wb') as f:
            pickle.dump(dict_obj, f, protocol = pickle.HIGHEST_PROTOCOL)


    def loadModel(model_name):
        '''
        This function takes in a model name and loads the model from the stored model file.

        Parameters:
        model_name (str) : The name of the model to load.

        Returns:
        loaded_model (keras model) : The model object of the model with name model_name.
        '''
        json_file = open('Models/{}.json'.format(model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("Models/{}.h5".format(model_name))
        return loaded_model


    def loadDict(dict_name):
        '''
        This function takes in a dictionary name and loads the dictionary from the stored dictionary file.

        Parameters:
        dict_name (str) : The name of dictionary to load.

        Returns:
        dict_obj (dict) : The dictionary object of the dictionary with name dict_name.
        '''
        with open('Models/{}.pickle'.format(dict_name), 'rb') as file:
            dict_obj = pickle.load(file)
        return dict_obj
