
def set_attributes(obj, attr_dict):
    for key, val in attr_dict.items():
        if hasattr(obj, key):
            setattr(obj, key, val)
