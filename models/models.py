models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
        
    return decorator


def make(model_spec, load_model=False):
    print("Making model:", model_spec["name"])
    print("Available models:", models.keys())

    model = models[model_spec["name"]](**model_spec["args"])

    if load_model:
        if "sd" not in model_spec:
            print("WARNING: no 'sd' found in model_spec, so weights were not loaded.")
            return model

        sd = model_spec["sd"]

        if isinstance(model, (tuple, list)):
            print("Model returned tuple/list. Loading weights into first model.")
            model_g = model[0]
            model_g.load_state_dict(sd)
            model = (model_g, *model[1:])
            print("LOADED weights into first model.")
        else:
            print("Model returned single model. Loading weights into it.")
            model.load_state_dict(sd)
            print("LOADED weights into single model.")

    return model
