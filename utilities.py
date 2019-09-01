import pickle

class Utilities:
    def saveModel(model, model_name):
        model_json = model.to_json()
        with open('Models/{}.json'.format(model_name), 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('Models/{}.h5'.format(model_name))

    def saveDict(dict_obj, dict_name):
        with open('Models/{}.pickle'.format(dict_name), 'wb') as f:
            pickle.dump(dict_obj, f, protocol = pickle.HIGHEST_PROTOCOL)
