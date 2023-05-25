import json
import os
import csv

class PredictionToJsonConverter:
    def __init__(self, path_to_labels, path_to_predictions):
        self.path_to_predictions = path_to_predictions
        self.path_to_labels = path_to_labels
    
    def convert_to_json(self):
        
        predictions = self.read_predictions_from_file(self.path_to_predictions)
        print(len(predictions))
        couples = self.read_labels_cuples(self.path_to_labels)
        print(len(couples))
        
        result = {}
        
        for couple in couples:
            name, spec = couple
            value = predictions.pop(0)
            
            if name not in result:
                result[name] = {}
            
            result[name][spec] = value
        
        # Specify the file path where you want to save the JSON data
        file_path = "./"+self.path_to_predictions+".p"

        # Save the JSON data to the file
        with open(file_path, 'w') as file:
            json.dump(result, file)
        
        return json.dumps(result, indent=4)

    def read_predictions_from_file(self, file_path):
        values = []
        with open(file_path, 'r') as file:
            data = file.read()
            data = data.replace("[", "").replace("]", ",")
            values = []
            for value in data.split(","):
                if value != '': 
                    values.append(float(value))
           
        return values

    def read_labels_cuples(self, file_path):
        couples = []
        skiped_first = False
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
           
            for row in reader:
                if skiped_first:
                    if len(row) >= 2:
                        couple = (row[0], row[1])
                        couples.append(couple)
                else:
                    skiped_first = True
        
        return couples


converter = PredictionToJsonConverter("./kpm_data/labels_dev.csv", "./prediction_dev_roberta-base")
json_output = converter.convert_to_json()


