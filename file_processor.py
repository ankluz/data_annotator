import json

def save_annotations(annotations, output_path):
    """Save annotations to JSON file"""
    print("Saving annotations (before conversion):", annotations)
    
    data = []
    for item in annotations:
        box = item[0]
        sequences = item[1]
        symbol = item[2] if len(item) > 2 else ""
        
        # Convert box coordinates to the correct format
        box_data = {
            'box': {
                'x': box[0],
                'y': box[1],
                'w': box[2],
                'h': box[3]
            },
            'sequences': [
                [{'x': float(point[0]), 'y': float(point[1])} for point in sequence]
                for sequence in sequences
            ],
            'symbol': symbol
        }
        data.append(box_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_annotations(input_path):
    """Load annotations from JSON file"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("Loaded JSON data:", data)
        
        annotations = []
        for item in data:
            # Load coordinates in original format
            box = (
                float(item['box']['x']),
                float(item['box']['y']),
                float(item['box']['w']),
                float(item['box']['h'])
            )
            sequences = [
                [(float(point['x']), float(point['y'])) for point in sequence]
                for sequence in item['sequences']
            ]
            symbol = item.get('symbol', "")  # Use get for backward compatibility
            annotations.append((box, sequences, symbol))
        
        return annotations
    except FileNotFoundError:
        print(f"Warning: File {input_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Warning: File {input_path} is not valid JSON: {e}")
        return []
    except KeyError as e:
        print(f"Warning: Invalid data format in {input_path}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error while loading annotations: {e}")
        return []