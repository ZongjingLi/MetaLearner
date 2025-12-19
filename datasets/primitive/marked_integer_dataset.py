import torch
import numpy as np
import numpy.random as npr
import cv2
from helchriss.utils.data import FilterableDatasetView, FilterableDatasetUnwrapped
from helchriss.utils.collate import VarLengthCollateV2
from torch.utils.data import DataLoader

# Constants for all possible sprite types
g_numbers_to_display = list(range(3))  # 0-9
g_shapes_index_to_name = {0: 'circle', 1: 'triangle', 2: 'rectangle'}
g_colors_index_to_name = {0: 'red', 1: 'green', 2: 'blue'}


def create_sprite(object_size: int = 32, sprite_type: str = 'number'):
    """
    Creates an image with either a numbered circle or a shape with a number,
    drawn in a colored sprite.
    
    Args:
        object_size: Size of the sprite
        sprite_type: 'number' or 'shape'
        
    Returns:
        Tuple of (image, sprite_info)
    """
    canvas_size = (object_size, object_size)  # h x w
    canvas = np.zeros(canvas_size + (3,), dtype=np.uint8)
    
    # Choose a random number 0-9 (all sprites have a number)
    number = npr.choice(g_numbers_to_display)
    
    # Choose a random color for both types
    color_idx = npr.randint(0, 3)  # 0: red, 1: green, 2: blue
    color_name = g_colors_index_to_name[color_idx]
    
    # Set the color in BGR format
    if color_idx == 0: color = (37, 37, 173) #color = [0, 0, 255]#  # Red (BGR format)
    elif color_idx == 1: color = color = (17, 215, 99)#[0, 255, 0]#(61, 156, 55)  # Green
    else: color = (165, 76, 49)#  # Blue
    
    # Font settings for text (used for both types)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text = str(number)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    if sprite_type == 'number':
        # Create a colored circular background for the sprite
        radius = int(object_size * 0.4)
        center = (object_size // 2, object_size // 2)
        canvas = cv2.circle(canvas, center, radius, color, -1)
        
        # Calculate text position to center it
        text_x = (object_size - text_width) // 2
        text_y = (object_size + text_height) // 2
        
        # draw the text on the canvas (white color)
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # return the image and sprite info
        sprite_info = {
            'type': 'number',
            'number': number, 
            'color': color_name,
            'shape': 'circle'
        }
        
    else:  # shape
        # Choose a random shape
        shape_idx = npr.randint(0, 3)  # 0: circle, 1: triangle, 2: rectangle
        shape_name = g_shapes_index_to_name[shape_idx]
        
        # Draw the shape
        if shape_idx == 0:  # circle
            radius = int(object_size * 0.4)
            center = (object_size // 2, object_size // 2)
            canvas = cv2.circle(canvas, center, radius, color, -1)
            
            # Calculate text position to center it
            text_x = (object_size - text_width) // 2
            text_y = (object_size + text_height) // 2
            
        elif shape_idx == 1:  # triangle
            pts = np.array([
                [object_size // 2, object_size // 4],
                [object_size // 4, object_size * 3 // 4],
                [object_size * 3 // 4, object_size * 3 // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)
            
            # For triangle, position text a bit lower
            text_x = (object_size - text_width) // 2
            text_y = (object_size + text_height) // 2 + int(object_size * 0.1)
            
        else:  # rectangle
            pts = np.array([
                [object_size // 4, object_size // 4],
                [object_size // 4, object_size * 3 // 4],
                [object_size * 3 // 4, object_size * 3 // 4],
                [object_size * 3 // 4, object_size // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)
            
            # For rectangle, center the text
            text_x = (object_size - text_width) // 2
            text_y = (object_size + text_height) // 2
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        # Return the image and sprite info
        sprite_info = {
            'type': 'shape',
            'shape': shape_name, 
            'color': color_name,
            'number': number
        }
    
    return canvas, sprite_info


def create_mixed_sprites3(object_size: int = 32, sprite_types=None):
    """
    Creates an image with 3 sprites, which can be either numbered circles or shapes with numbers.
    
    Args:
        object_size: Size of each sprite
        sprite_types: List of sprite types to use ('number' or 'shape'). 
                     If None, randomly choose for each position.
        
    Returns:
        Tuple of (image, list of sprite_info)
    """
    canvas_size = (object_size, object_size * 3)  # h x w
    canvas = np.zeros(canvas_size + (3,), dtype=np.uint8)
    sprites = list()

    # if sprite_types not specified, randomly choose for each position
    if sprite_types is None:
        sprite_types = [npr.choice(['number', 'shape']) for _ in range(3)]
        
    for i in range(3):
        # Create the sprite
        sprite_type = sprite_types[i]
        sprite_img, sprite_info = create_sprite(object_size, sprite_type)
        sprites.append(sprite_info)
        
        # Place the sprite on the canvas
        x_offset = i * object_size
        canvas[:, x_offset:x_offset+object_size, :] = sprite_img

    return canvas, sprites


def _gen_random_label():
    """Generate a random label for questions"""
    # This covers both number and shape types
    label_types = [
        #'number',   # For both sprites types (they all have numbers)
        'shape',    # For shape sprites
        'color'     # For both
    ]
    
    label_type = npr.choice(label_types)
    
    if label_type == 'number':
        number = npr.choice(g_numbers_to_display)
        return f'number {number}', 'number', str(number)
    elif label_type == 'shape':
        shape = npr.choice(list(g_shapes_index_to_name.values()))
        return f'{shape}', 'shape', shape
    else:  # color
        color = npr.choice(list(g_colors_index_to_name.values()))
        return f'{color} sprite', 'color', color


# Changes to _gen_random_question function for boolean questions
def _gen_random_question(sprites, arity: int):
    """Generate a random boolean question about the sprites with shorter syntax"""
    if arity == 1:
        label, label_type, label_value = _gen_random_label()

        # Check if the label exists in any sprite
        if label_type == 'number':
            answer = any(sprite['number'] == int(label_value) for sprite in sprites)
        elif label_type == 'shape':
            answer = any(sprite['type'] == 'shape' and sprite['shape'] == label_value for sprite in sprites)
        else:  # color
            answer = any(sprite['color'] == label_value for sprite in sprites)
            
        # Shorter query format
        question = f'{label} exists'
        program = f'exists:Logic( filter:Logic(scene:Objects(), {label_value}:Objects ) )'
        return question, program, answer
        
    else:  # arity == 2
        label1, label_type1, label_value1 = _gen_random_label()
        label2, label_type2, label_value2 = _gen_random_label()
        
        relation = npr.choice(['left', 'right'])
        answer = False

        if relation == 'left':
            indices = [(1, 0), (2, 1), (2, 0)]
            
        else:
            indices = [(0, 1), (1, 2), (0, 2)]
            
            
        for i, j in indices:
            # Check if first sprite matches label1
            if (label_type1 == 'number' and sprites[i]['number'] == int(label_value1)) or \
               (label_type1 == 'shape' and sprites[i]['type'] == 'shape' and sprites[i]['shape'] == label_value1) or \
               (label_type1 == 'color' and sprites[i]['color'] == label_value1):
                
                # Check if second sprite matches label2
                if (label_type2 == 'number' and sprites[j]['number'] == int(label_value2)) or \
                   (label_type2 == 'shape' and sprites[j]['type'] == 'shape' and sprites[j]['shape'] == label_value2) or \
                   (label_type2 == 'color' and sprites[j]['color'] == label_value2):
                    answer = True
                    break

        # Shorter query format
        question = f'{label1} {relation} of {label2}'
        #print(question)

        program = f'exists(Object, lambda x: exists(Object, lambda y: {label_value1}(x) and {relation}(x, y) and {label_value2}(y)))'
        program = f"exists:Logic(relate:Logic(filter:Logic(scene:Objects(),{label_value1}:Objects),filter:Logic(scene:Objects(),{label_value2}:Objects),{relation}:Objects))"
        return question, program, answer


# Changes to _gen_arithmetic_question function

def _gen_arithmetic_question(sprites):
    """Generate arithmetic questions about the sprites"""
    question_types = [
        'sum_color',        # "What is the sum of red numbers?"
        'add_value',        # "What is the red object plus 1?"
        'max_color',        # "What is the maximum blue number?"
        'min_color',        # "What is the minimum green number?"
        'diff_colors',      # "What is the difference between the red number and the blue number?"
        'product_color',    # "What is the product of green numbers?"
        'avg_color',        # "What is the average of red numbers?"
        'count_shape',      # "How many triangles are there?"
        'count_color',      # "How many blue objects are there?"
        'sum_shape',        # "What is the sum of triangle numbers?"
        'sum_red_blue',     # "What is the sum of red and blue numbers?"
        'max_number_shape', # "What is the maximum number on a triangle?"
    ]
    question_types = [
        'count_shape',      # "How many triangles are there?"
        'count_color',      # "How many blue objects are there?"
        'max_color', # "What is the maximum number on a triangle?"
        'max'
    ]
    
    question_type = npr.choice(question_types)
    color_choices = ['red', 'green', 'blue']
    shape_choices = ['circle', 'triangle', 'rectangle']
    
    # Handle questions about counting or summing specific shapes
    if question_type == 'count_shape':
        shape = npr.choice(shape_choices)
        # Count all sprites of the specified shape
        count = sum(1 for sprite in sprites if sprite['type'] == 'shape' and sprite['shape'] == shape)
        
        question = f"How many {shape}s are there"
        program = f"count:Logic(filter:Logic(scene:Objects(), {shape}:Objects ))"

        return question, program, count
        
    elif question_type == 'count_color':
        color = npr.choice(color_choices)
        # Count all sprites of the specified color
        count = sum(1 for sprite in sprites if sprite['color'] == color)
        
        question = f"How many {color} objects are there"
        program = f"count:Logic(filter:Logic(scene:Objects(), {color}:Objects ))"
        
        return question, program, count
        
    elif question_type == 'sum_shape':
        shape = npr.choice(shape_choices)
        # Get all sprites of the specified shape
        shape_sprites = [sprite for sprite in sprites if sprite['type'] == 'shape' and sprite['shape'] == shape]
        
        # Calculate the sum of their numbers
        if shape_sprites:
            answer = sum(sprite['number'] for sprite in shape_sprites)
        else:
            answer = 0
            
        question = f"What is the sum of {shape} numbers"
        program = f"sum(filter(Object, lambda x: {shape}(x)), lambda x: number(x))"
        
        return question, program, answer
    
    # Handle color-based arithmetic questions
    elif question_type == 'sum_color':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # Calculate the sum of numbers
        if color_sprites:
            answer = sum(sprite['number'] for sprite in color_sprites)
        else:
            answer = 0
            
        question = f"What is the sum of {color} numbers"
        program = f"sum(filter(Object, lambda x: {color}(x)), lambda x: number(x))"
        
        return question, program, answer
        
    elif question_type == 'add_value':
        color = npr.choice(color_choices)
        # Find objects of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # If there are no objects of this color, try another color
        if not color_sprites:
            # Try to find a color that exists in the sprites
            for alt_color in color_choices:
                color_sprites = [sprite for sprite in sprites if sprite['color'] == alt_color]
                if color_sprites:
                    color = alt_color
                    break
        
        # Pick a random value to add
        value = npr.choice([0, 1, 2])
        
        # Get the first sprite of the chosen color if it exists
        if color_sprites:
            sprite_number = color_sprites[0]['number']
            answer = sprite_number + value
        else:
            # Fallback if somehow no color is found
            answer = value
            sprite_number = 0
            
        question = f"What is the {color} object plus {value}"
        program = f"add(filter(Object, lambda x: {color}(x))[0].number, {value})"
        
        return question, program, answer
        
    elif question_type == 'max_color':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # Find the maximum number
        if color_sprites:
            answer = max(sprite['number'] for sprite in color_sprites)
        else:
            # If no sprites of this color, answer is -1 (indicating no such sprite)
            answer = 0
            
        question = f"What is the maximum {color} number"
        program = f"max:Integer(filter:Logic(scene:Objects(), {color}:Objects ) )"
        
        return question, program, answer

    elif question_type == 'max':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites ]
        
        # Find the maximum number
        if color_sprites:
            answer = max(sprite['number'] for sprite in color_sprites)
        else:
            # If no sprites of this color, answer is -1 (indicating no such sprite)
            answer = 0
            
        question = f"What is the maximum number"
        program = f"max:Integer(scene:Objects() )"
        
        return question, program, answer
        
    elif question_type == 'min_color':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # Find the minimum number
        if color_sprites:
            answer = min(sprite['number'] for sprite in color_sprites)
        else:
            # If no sprites of this color, answer is -1 (indicating no such sprite)
            answer = -1
            
        question = f"What is the minimum {color} number"
        program = f"min(filter(Object, lambda x: {color}(x)), lambda x: number(x))"
        
        return question, program, answer
        
    elif question_type == 'diff_colors':
        color1, color2 = npr.choice(color_choices, size=2, replace=False)
        # Get sprites of each color
        color1_sprites = [sprite for sprite in sprites if sprite['color'] == color1]
        color2_sprites = [sprite for sprite in sprites if sprite['color'] == color2]
        
        # Calculate difference between first sprites of each color
        if color1_sprites and color2_sprites:
            answer = color1_sprites[0]['number'] - color2_sprites[0]['number']
        else:
            # If either color doesn't exist, answer is 0
            answer = 0
            
        question = f"difference between the {color1} number and the {color2} number"
        program = f"subtract(filter(Object, lambda x: {color1}(x))[0].number, filter(Object, lambda x: {color2}(x))[0].number)"
        
        return question, program, answer
        
    elif question_type == 'product_color':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # Calculate the product
        if color_sprites:
            answer = 1
            for sprite in color_sprites:
                answer *= sprite['number']
        else:
            # If no sprites of this color, answer is 0
            answer = 0
            
        question = f"the product of {color} numbers"
        program = f"product(filter(Object, lambda x: {color}(x)), lambda x: number(x))"
        
        return question, program, answer
        
    elif question_type == 'avg_color':
        color = npr.choice(color_choices)
        # Get all sprites of the specified color
        color_sprites = [sprite for sprite in sprites if sprite['color'] == color]
        
        # Calculate the average
        if color_sprites:
            avg = sum(sprite['number'] for sprite in color_sprites) / len(color_sprites)
            # Round to 1 decimal place for simplicity
            answer = round(avg, 1)
        else:
            # If no sprites of this color, answer is 0
            answer = 0
            
        question = f"the average of {color} numbers"
        program = f"average(filter(Object, lambda x: {color}(x)), lambda x: number(x))"
        
        return question, program, answer
        
    elif question_type == 'sum_red_blue':
        # Get all red and blue sprites
        red_sprites = [sprite for sprite in sprites if sprite['color'] == 'red']
        blue_sprites = [sprite for sprite in sprites if sprite['color'] == 'blue']
        
        # Calculate sum of both colors
        red_sum = sum(sprite['number'] for sprite in red_sprites) if red_sprites else 0
        blue_sum = sum(sprite['number'] for sprite in blue_sprites) if blue_sprites else 0
        answer = red_sum + blue_sum
        
        question = f"the sum of red and blue numbers"
        program = f"sum(filter(Object, lambda x: red(x) or blue(x)), lambda x: number(x))"
        
        return question, program, answer
        
    elif question_type == 'max_number_shape':
        shape = npr.choice(shape_choices)
        # Get all sprites of the specified shape
        shape_sprites = [sprite for sprite in sprites if (sprite['type'] == 'shape' and sprite['shape'] == shape) or 
                                                        (sprite['type'] == 'number' and sprite['shape'] == shape)]
        
        # Find the maximum number
        if shape_sprites:
            answer = max(sprite['number'] for sprite in shape_sprites)
        else:
            # If no sprites of this shape, answer is -1
            answer = -1
            
        question = f"What is the maximum number on a {shape}"
        program = f"max:Integer(filter:Logic(scene:Objects(), {shape}))"

        return question, program, answer


def gen_mixed_sprites3_dataset(dataset_size, percent = 1.0, unary = 1.0):
    """Generate the complete dataset of mixed sprites (numbers and shapes)"""
    images, sprites_info, questions, programs, answers = list(), list(), list(), list(), list()
    
    for i in range(dataset_size):
        # Randomly decide if we want all one type or mixed types
        if npr.rand() < 1.0:  # 100% chance of being all one type
            sprite_type = "shape"#npr.choice(['number', 'shape'])
            sprite_types = [sprite_type, sprite_type, sprite_type]
        else:
            # Mixed types
            sprite_types = [npr.choice(['number', 'shape']) for _ in range(3)]
        
        image, sprites = create_mixed_sprites3(sprite_types=sprite_types)
        images.append(image)
        sprites_info.append(sprites)

        # Choose a question type: boolean (original) or arithmetic (new)
        question_type = npr.choice(['boolean', 'arithmetic'], p = [percent, 1 - percent]) # 'arithmetic'
        
        if question_type == 'boolean':
            # Original boolean questions
            arity = npr.choice([1, 2], p = [unary, 1- unary]) # 2
            answer = npr.choice([True, False])
            for trials in range(50):
                question, logical_form, pred_answer = _gen_random_question(sprites, arity)
                if pred_answer == answer:
                    break
                    
            questions.append(question)
            programs.append(logical_form)
            answers.append(pred_answer)
        else:
            # New arithmetic questions
            question, logical_form, answer = _gen_arithmetic_question(sprites)
            #print(question)
            questions.append(question)
            programs.append(logical_form)
            answers.append(answer)

    return dict(
        images=images, 
        sprites=sprites_info, 
        questions=questions, 
        programs=programs, 
        answers=answers
    )


class MixedSprites3DatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, dataset_size, regular_percent = 1.0, unary = 1.0):
        super().__init__()
        self.data = gen_mixed_sprites3_dataset(dataset_size, regular_percent, unary = unary)

    def _get_metainfo(self, index):
        return {
            'query': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': self.data['answers'][index],  # Can be bool or numeric
            'program' : self.data['programs'][index],
            'question_length': len(self.data['questions'][index].split()),
            'question_type': 'boolean' if isinstance(self.data['answers'][index], bool) else 'arithmetic'
        }

    def __getitem__(self, index):
        return {
            'image': _to_image(self.data['images'][index]),
            'query': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': self.data['answers'][index],  # Can be bool or numeric
            'sprites': self.data['sprites'][index],
            'grounding' : {"image" : _to_image(self.data['images'][index])}
        }

    def __len__(self):
        return len(self.data['images'])


def _to_image(image):
    """Convert image to PyTorch tensor format (with RGB ↔ BGR swap)"""
    # Swap R and B channels (channel dimension is index 2 for HWC format)
    image = image[..., ::-1]  # Reverse last dimension (RGB → BGR)
    # Original processing steps
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = image / 255.0
    image = image.astype(np.float32)
    #image = (image - 0.5) * 2
    return torch.tensor(image)

class MixedSprites3DatasetFilterableView(FilterableDatasetView):
    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> DataLoader:
        collate_guide = {
            'query': 'skip',
            'program': 'skip',
            'answer': 'skip',
            'sprites': 'skip'
        }
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )

    def filter_question_length(self, length: int) -> 'MixedSprites3DatasetFilterableView':
        """Filter dataset based on question length"""
        def filt(meta):
            return meta['question_length'] <= length
        return self.filter(filt, f'filter-qlength[{length}]')
    
    def filter_by_answer(self, answer) -> 'MixedSprites3DatasetFilterableView':
        """Filter dataset based on answer (can be boolean or numeric)"""
        def filt(meta):
            return meta['answer'] == answer
        return self.filter(filt, f'filter-answer[{answer}]')
    
    def filter_by_question_type(self, question_type: str) -> 'MixedSprites3DatasetFilterableView':
        """Filter dataset based on question type (boolean or arithmetic)"""
        def filt(meta):
            return meta['question_type'] == question_type
        return self.filter(filt, f'filter-qtype[{question_type}]')
    
    def filter_by_sprite_types(self, has_numbers: bool = None, has_shapes: bool = None) -> 'MixedSprites3DatasetFilterableView':
        """Filter dataset based on presence of number or shape sprites"""
        def filt(meta):
            sample = self._dataset[meta['index']]
            sprites = sample['sprites']
            
            has_number_sprites = any(sprite['type'] == 'number' for sprite in sprites)
            has_shape_sprites = any(sprite['type'] == 'shape' for sprite in sprites)
            
            if has_numbers is not None and has_shapes is not None:
                return has_number_sprites == has_numbers and has_shape_sprites == has_shapes
            elif has_numbers is not None:
                return has_number_sprites == has_numbers
            elif has_shapes is not None:
                return has_shape_sprites == has_shapes
            else:
                return True
                
        return self.filter(filt, f'filter-sprite-types[numbers={has_numbers},shapes={has_shapes}]')


def MixedSprites3Dataset(dataset_size, p, unary = 1.0) -> MixedSprites3DatasetFilterableView:
    """
    Create a filterable dataset of images with mixed sprites (numbers and shapes).
    
    Each image contains 3 sprites, where each sprite can be:
    1. A number (0-9) displayed in the center of a colored circle (red, green, or blue)
    2. A shape (circle, triangle, or rectangle) filled with a color (red, green, or blue)
       and also containing a randomly generated number (0-9)
    
    All sprites (both number-type and shape-type) have a number for arithmetic operations.
    
    The dataset includes two types of questions:
    1. Boolean questions (yes/no):
       - "Is there a red sprite?"
       - "Is there a number 5 to the left of a blue sprite?"
       - "Is there a triangle to the right of a green sprite?"
    
    2. Arithmetic questions (numeric answers):
       - "What is the sum of red numbers?"
       - "What is the red object plus 1.0?"
       - "What is the maximum blue number?"
       - "What is the minimum green number?"
       - "What is the difference between the red number and the blue number?"
       - "What is the product of green numbers?"
       - "What is the average of red numbers?"
       - "How many triangles are there?"
       - "How many blue objects are there?"
       - "What is the sum of triangle numbers?"
       - "What is the sum of red and blue numbers?"
       - "What is the maximum number on a triangle?"
    
    Usage:
        dataset = MixedSprites3Dataset(1000)  # 1000 data points
        
        # Filter the dataset
        short_questions = dataset.filter_question_length(10)
        true_answers = dataset.filter_by_answer(True)
        arithmetic_questions = dataset.filter_by_question_type('arithmetic')
        only_numbers = dataset.filter_by_sprite_types(has_numbers=True, has_shapes=False)
        
        # Create dataloader
        dataloader = dataset.make_dataloader(
            batch_size=32, shuffle=True, drop_last=False, nr_workers=4
        )
    """
    return MixedSprites3DatasetFilterableView(MixedSprites3DatasetUnwrapped(dataset_size, p, unary))




import os
import json
import cv2
import numpy as np

def save(dataset_size: int, save_root: str, regular_percent: float = 1.0):
    """
    Generates MixedSprites3 dataset and saves it to a structured folder hierarchy.
    Preserves original image resolution (saved as PNG) and human-readable formats.
    
    Args:
        dataset_size: Number of samples to generate
        save_root: Root directory to save dataset (e.g., "./mixed_sprites_dataset")
        regular_percent: Percentage of boolean questions (0.0-1.0)
    """
    # Step 1: Generate raw dataset data
    raw_data = gen_mixed_sprites3_dataset(dataset_size, regular_percent)
    images = raw_data['images']
    queries = raw_data['questions']
    programs = raw_data['programs']
    answers = [str(ans) for ans in raw_data['answers']]
    sprites_info = {}#raw_data['sprites']

    # Step 2: Create folder structure
    img_dir = os.path.join(save_root, "imgs")
    query_dir = os.path.join(save_root, "queries")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)

    # Step 3: Save original images (PNG format, no modification)
    for idx, img in enumerate(images):
        # Convert BGR (cv2 default) to RGB for correct color display
        img_rgb = img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_path = os.path.join(img_dir, f"{idx}.png")
        cv2.imwrite(img_path, img_rgb)

    # Step 4: Save queries (one text file per sample)
    for idx, query in enumerate(queries):
        query_path = os.path.join(query_dir, f"{idx}.txt")
        with open(query_path, "w", encoding="utf-8") as f:
            f.write(query.strip())

    # Step 5: Save structured metadata (JSON/TXT for easy access)
    # - Programs: one line per sample
    with open(os.path.join(save_root, "programs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(programs))
    
    # - Answers: JSON (supports bool/numeric values)
    with open(os.path.join(save_root, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)
    
    # - Sprite info: JSON of detailed sprite metadata
    with open(os.path.join(save_root, "sprites_info.json"), "w", encoding="utf-8") as f:
        json.dump(sprites_info, f, indent=2)
    
    # - Dataset stats: human-readable info
    bool_count = sum(isinstance(ans, bool) for ans in answers)
    arith_count = dataset_size - bool_count
    with open(os.path.join(save_root, "dataset_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset Size: {dataset_size}\n")
        f.write(f"Boolean Questions: {bool_count} ({bool_count/dataset_size*100:.1f}%)\n")
        f.write(f"Arithmetic Questions: {arith_count} ({arith_count/dataset_size*100:.1f}%)\n")
        f.write(f"Regular Percent (Boolean Ratio): {regular_percent}\n")

    print(f"Dataset saved successfully to: {save_root}")
    #print(f"Folder structure:\n- {img_dir} ({len(images)} PNGs)\n- {query_dir} ({len(queries)} TXTs)\n- Metadata files: programs.txt, answers.json, sprites_info.json, dataset_info.txt")


if __name__ == "__main__":
    save(1024, "/Users/sunyiqi/Documents/Datasets", 0.5)