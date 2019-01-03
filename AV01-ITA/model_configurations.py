class ModelDefinition:    
    def __init_(self):
        pass

    model_definitions = {
        0:{
            "batch_size" : 128,
            "epochs": 10,
            "iterations" : 1,
            "layer1kernel" : 3,
            "layer2kernel" :3,
            "layer1pad" : "same",
            "layer2pad" : "same",
            "layer1filters":24,
            "layer2filters":64,
            "layer1pool":2,
            "layer2pool":2,
            "layer1dense":128,
            "layer2dense":128,
            "drop1":0.3,
            "drop2":0.4,
            "drop3":0.5
        }
    }