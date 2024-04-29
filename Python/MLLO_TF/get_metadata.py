

from datetime import datetime
from mlloutil import get_manual_input, get_config_value_type, to_pascal_case, to_str, load_model, is_all_dicts_identical, to_json, json_validation


def get_layer_dimension(layer):
    # Get layer dimension from each layer type
    layer_type = layer.__class__.__name__
    # Use built-in functions for common layer types
    if layer_type in {'Conv2D', 'Conv3D', 'Conv1D'}:
        dimensions =  [layer.get_config()["filters"]] + list(layer.get_config()['kernel_size'])
    elif layer_type == 'Dense':
        dimensions = [layer.get_config()['units']] 
    elif layer_type == 'LSTM':
        dimensions = list(layer.get_config()['units'])
    elif layer_type == 'Embedding':
        dimensions = [layer.get_config()['input_dim']]
    elif layer_type == 'Flatten':
        dimensions = 1 
    elif layer_type == 'MaxPooling2D' or layer_type == 'MaxPooling3D':
        dimensions = list(layer.get_config()['pool_size'])
    elif layer_type == 'AveragePooling2D' or layer_type == 'AveragePooling3D':
        dimensions = list(layer.get_config()['pool_size'])

    # Check for potentially nested configurations like Recurrent layers
    elif 'rnn' in layer_type.lower():
        try:
            dimensions = layer['layers'][0]['units']
        except KeyError:
            dimensions = None  # Handle nested layers recursively

    # Handle unsupported layer types
    else:
        dimensions = None
    
    return dimensions

def get_layers(model):
    """
    Retrieve layers from KerasTensoflow model.
    Return dictionary that has layer index, input layers, neural network layers, 
    and others (e.g. activation, initializer, etc.)
    """
    model_layers = model.layers
    input_layers = []
    nn_layers = []
    other_layers = []
    layer_index_dict = {}
    layer_index = 0

    for layer in model_layers:
        if layer.__class__.__name__ in {'InputLayer'}:
            input_layers.append(layer)
        elif layer.__class__.__name__ in {'transform_features_layer', "Dropout", "Activation"}:
            other_layers.append(layer)
        else:
            nn_layers.append(layer)


        if layer.__class__.__name__.startswith("Dropout"):
            layer_index_dict[layer.get_config()['name']] = layer_index - 1 if layer_index > 0 else layer_index

        else:
            layer_index_dict[layer.get_config()['name']] = layer_index
            layer_index += 1


    return {"layer_index":layer_index_dict,'input_layers':input_layers, 'nn_layers':nn_layers, 'other_layers':other_layers}  

def get_parameters_layer(model):
    model_layers = get_layers(model)['nn_layers']
    parameter_layers_list = []
    layers_dimensions = {}
    layer_uid = 0

    try:
        for layer in model_layers:
            parameter_layer = {"parameter_layer":{}}
            activation_function = {}
            layer_config_setting = []
            layer_config_count = 0
            for layerconfig in layer.get_config():
                if layerconfig not in ["name" ,"batch_input_shape", "dtype", "kernel_initializer","bias_initializer", "embeddings_initializer", "activation"]:
                    config_value = to_str(layer.get_config()[layerconfig])
                    configuration_setting_dict = {
                        "configuration_setting": {
                            "uid": f"LayerConfigurationSetting{layer_uid}{layer_config_count}",
                            "configuration_setting_type": to_pascal_case(layerconfig),
                            "configuration_setting_value_type":get_config_value_type(config_value),
                            "configuration_setting_value": config_value,
                        }
                    }
                    layer_config_setting.append(configuration_setting_dict)
                    layer_config_count +=1
                    
                #get activation_function
                if layerconfig in ["activation"]:
                    activation_config_setting = []
                    if isinstance(layer.get_config()[layerconfig], str):
                        config_value = to_str(layer.get_config()[layerconfig])
                        activation_function = {
                            "uid":f"ActivationFuction{layer_uid}",
                            "activation_function_type": layer.get_config()[layerconfig]
                        }


                    if isinstance(layer.get_config()[layerconfig], dict):
                        ac_uid = 0
                        for activation_config in layer.get_config()[layerconfig]:
                            if activation_config not in ["class_name"]:
                                
                                config_value = to_str(layer.get_config()[layerconfig][activation_config])
                                configuration_setting_dict = {
                                    "configuration_setting": {
                                        "uid": f"ActivationConfigurationSetting{layer_uid}{ac_uid}",
                                        "configuration_setting_type": to_pascal_case(activation_config),
                                        "configuration_setting_value_type":get_config_value_type(config_value),
                                        "configuration_setting_value": config_value,
                                    }
                                }
                                ac_uid += 1
                                activation_config_setting.append(configuration_setting_dict)

                        config_value = to_str(layer.get_config()[layerconfig])
                        activation_function = {
                            "uid":f"ActivationFuction{layer_uid}",
                            "activation_function_type": to_str(layer.get_config()[layerconfig]["class_name"]),
                            "configuration_setting":activation_config_setting
                        }
                    parameter_layer["parameter_layer"]['activation_function'] = activation_function
                else:
                    continue
                    
            layer_name = layer.name
            layer_type = layer.__class__.__name__

            dimensions = get_layer_dimension(layer)
            if dimensions:
                layers_dimensions[layer_name] = dimensions
            
            parameter_layer["parameter_layer"]["uid"] = f"ParameterLayer{layer_uid}"
            parameter_layer["parameter_layer"]["layer_index"] = layer_uid
            parameter_layer["parameter_layer"]["layer_dimension"] = str(dimensions)
            parameter_layer["parameter_layer"]["layer_input_dimension"] = str([i for i in model_layers[layer_uid].input_shape if i is not None])
            parameter_layer["parameter_layer"]["layer_type "] = f"{layer_type}"
            parameter_layer["parameter_layer"]["layer_configuration"] = layer_config_setting
            

            parameter_layers_list.append(parameter_layer)
            
            layer_uid += 1
            
    except Exception as error:
            
            print("An error occurred:", error)
            print(f'error at {layer.name} layer')

    return parameter_layers_list

def get_regularizer(model):
    model_layers = get_layers(model)['other_layers']
    layer_index = get_layers(model)['layer_index']
    regularizer_list= []
    regularizer_configuration_list = []

    regularizer_uid = 0
    regularizer_config_uid = 0
    configuration_setting_dict = {}
    
    for layer in model_layers:
        for i in layer.get_config():
            if i not in ["name"]:
                print(i)
                configuration_setting_dict = {
                    "configuration_setting": {
                        "uid": f"RegularizerConfigurationSetting{regularizer_config_uid}",
                        "configuration_setting_type": to_pascal_case(i),
                        "configuration_setting_value_type": get_config_value_type(layer.get_config()[i]),
                        "configuration_setting_value": to_str(layer.get_config()[i]),
                    }
                }
                regularizer_configuration_list.append(configuration_setting_dict)
                regularizer_config_uid += 1
        

            if layer.__class__.__name__ in ["Dropout"]:
                regularizer_dict = {
                    "uid": f"Regularizer{layer_index[layer.get_config()['name']]}{regularizer_uid}",
                    "regularizer_type": layer.__class__.__name__,
                    "layer_index": layer_index[layer.get_config()['name']],
                    "regularizer_configuration": regularizer_configuration_list
                    
                }
            regularizer_uid +=1
            regularizer_list.append(regularizer_dict)
        
    return regularizer_list
    
def get_training_config(model):
    model_optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.get_config()['name']
    model_optimizer_config.pop('name')
    model_hyperparameter_config = {"Epoch": 10, "BatchSize": 128, "Verbose":1}
    training_config = {}
    hyperparameters_configuration_list = []
    optimizers_configuration_list = []
    hyperparameter_config_uid = 0
    optimizer_config_uid = 0

    for i in model_hyperparameter_config:
        configuration_setting_dict = {
            "configuration_setting": {
                "uid": f"HyperparameterConfigurationSetting{hyperparameter_config_uid}",
                "configuration_setting_type": to_pascal_case(i),
                "configuration_setting_value_type": get_config_value_type(model_hyperparameter_config[i]),
                "configuration_setting_value": to_str(model_hyperparameter_config[i])

            }
        }
        hyperparameters_configuration_list.append(configuration_setting_dict)
        hyperparameters = hyperparameters_configuration_list
        hyperparameter_config_uid += 1
    
    optimizer_uid = 0
    for i in model_optimizer_config:
        configuration_setting_dict = {
            "configuration_setting": {
                "uid": f"OptimizerConfigurationSetting{optimizer_uid}{optimizer_config_uid}",
                "configuration_setting_type": to_pascal_case(i),
                "configuration_setting_value_type": get_config_value_type(model_optimizer_config[i]),
                "configuration_setting_value": to_str(model_optimizer_config[i]),
            }
        }
        optimizer_config_uid +=1

        optimizers_configuration_list.append(configuration_setting_dict)

    optimizer = {
        "uid": f"Optimizers{optimizer_uid}",
        "optimizer_type": optimizer_type,
        "opimizer_configuration": optimizers_configuration_list
    }

    return hyperparameters, optimizer

def get_initializer(model):

    model_layers = get_layers(model)['nn_layers']
    layer_uid = 0
    initializers = ["kernel_initializer","bias_initializer"]
    initializers_results = {}
    initializer_list = []
    initializer_dict_list = []
    
    for initializer in initializers:
        initializer_list = []
            
        for layer in model_layers:
            if initializer in layer.get_config():
                initializer_config_dict = layer.get_config()[initializer]
                initializer_list.append(initializer_config_dict)
            else:
                continue
        whole_model, output = is_all_dicts_identical(initializer_list)
        initializers_results[initializer] = {'AllIdentical': whole_model, 'Data': output}
        if initializers_results[initializer]['AllIdentical'] == True:
            init_config_setting = []
            initializer_dict = {
                "initializer": {
                "uid": f"{to_pascal_case(initializer)}Initializer{layer_uid}",
                "initializer_type": to_str(initializers_results[initializer]["Data"]["class_name"]),
                "whole_model": True,
                "initializes_layer_index": layer_uid
                }
            }

            for initializer_config in initializers_results[initializer]['Data']['config']:
                config_uid = 0
                configuration_setting_dict = {
                        "uid": f"{to_pascal_case(initializer)}ConfigurationSetting{config_uid}",
                        "configuration_setting_type": to_pascal_case(initializer_config),
                        "configuration_setting_value_type":get_config_value_type(initializers_results[initializer]['Data']['config'][initializer_config]),
                        "configuration_setting_value": initializers_results[initializer]['Data']['config'][initializer_config]
                    }
                init_config_setting.append(configuration_setting_dict)
                config_uid += 1
                initializer_dict["initializer"]["initializer_configuration"] = init_config_setting

            initializer_dict_list.append(initializer_dict)

        else:
            init_uid = 0
            for i in range(len(model_layers)):
                for k in ["kernel_initializer", "bias_initializer"]:
                    if k in model_layers[i].get_config():     
                        init_config_setting = []
                        for j in model_layers[i].get_config()[k]["config"]:
                            config_uid = 0
                            configuration_setting_dict = {
                                "configuration_setting": {
                                    "uid": f"ConfigurationSetting{init_uid}{config_uid}",
                                    "configuration_setting_type": to_pascal_case(j),
                                    "configuration_setting_value_type": get_config_value_type(model_layers[i].get_config()[k]["config"][j]),
                                    "configuration_setting_value": to_str(model_layers[i].get_config()[k]["config"][j]),
                                }
                            }
                            init_config_setting.append(configuration_setting_dict)
                            config_uid += 1
                            
                        initializer = {
                        "initializer": {
                            "uid": f"{to_pascal_case(k)}Initializer{init_uid}",
                            "initializer_type": f"{model_layers[i].get_config()[k]['class_name']}",
                            "configuration_setting": init_config_setting,
                            "whole_model": False,
                            "initializes_layer_index": init_uid,
                            }
                        }
                        initializer_dict_list.append(initializer)
                    else:
                        continue

                init_uid += 1
    return initializer_dict_list

def get_meta_data(model_path, framework, *inputs):
    # if framework not in frameworks:
    #     raise ValueError("Invalid sim type. Expected one of: %s" % frameworks)
    model = load_model(model_path)
    date_now = datetime.strftime(datetime.now(), "%y%m%dt%H%M%S")
    if framework == "KerasTensorflow":
        path = "test"
        model_name = "mnist_conv"
        model_location = "mnist"
        hyperparameter_config, optimizer_config = get_training_config(model)
        regularizer = get_regularizer(model)
        manual_inputs = get_manual_input(inputs)
        parameters_layers = get_parameters_layer(model)
        input_parameters_layers = model.get_config()["layers"][0]["config"]
        input_dimension = [i for i in input_parameters_layers["batch_input_shape"] if i is not None]
        initializer = get_initializer(model)

        mllo = {
            "uid": f"ModelName{date_now}",
            "model_name": f"{model_name}",
            "model_type": to_pascal_case(f"{model.name}"),
            "model_framework": {
                "uid": f"{to_pascal_case(framework)}{date_now}",
                "framework_type": framework,
                "framework_version": f"{2.14}",
            },
            "model_location": f"{model_location}",
            "created_at": date_now,
            "created_for_project": "MLLOS",
            "model_input_requirements": {
                "uid":"InputRequirements0",
                "input_dimension": to_str(input_dimension),
                "input_datatype": to_pascal_case(input_parameters_layers["dtype"]),
            },
            "model_architecture": {
                "uid":"ModelArchitecture0",
                "parameters": parameters_layers},
            "training_configuration": {
                "uid":"TrainingConfiguration0",
                "hyperparameters": hyperparameter_config,
                "optimizer": optimizer_config,
                "model_initializers": initializer,
                "model_regularizers": regularizer
            },  
            "score": {
                "uid": "Score0",
                "score_value": "0.7994",
                "test_data": "mnist_nonoise"
            }
            
            
        }
    else:
        print("Framwork error")
        mllo = "Framwork error"
    return mllo



if __name__ == '__main__':

    model_path = input('input model path')
# schema_path = input('schema path')

    #model_path = 'toy_model'
    schema_path = 'ml6.json'

    mllo_dict = get_meta_data(model_path, framework='KerasTensorflow')
    mllo_json_path = to_json(mllo_dict)
    json_validation(schema_path, mllo_json_path)
