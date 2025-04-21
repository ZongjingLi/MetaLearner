import torch
import numpy as np
import numpy.random as npr
import cv2
from helchriss.utils.data import FilterableDatasetView, FilterableDatasetUnwrapped
from helchriss.utils.collate import VarLengthCollateV2
from torch.utils.data import DataLoader

g_shapes_index_to_name = {0: 'circle', 1: 'triangle', 2: 'rectangle'}
g_colors_index_to_name = {0: 'red', 1: 'green', 2: 'blue'}


def create_shapes3(object_size: int = 32):
    canvas_size = (object_size, object_size * 3)  # h x w
    canvas = np.zeros(canvas_size + (3, ), dtype=np.uint8)
    shapes = list()

    for i in range(3):
        shape = npr.randint(0, 3)  # 0: circle, 1: triangle, 2: rectangle
        color = npr.randint(0, 3)  # 0: red, 1: green, 2: blue

        shapes.append({'shape': g_shapes_index_to_name[shape], 'color': g_colors_index_to_name[color]})

        if color == 0:
            color = (0, 0, 200)
        elif color == 1:
            color = (0, 200, 0)
        else:
            color = (200, 0, 0)

        if shape == 0:
            radius = int(object_size * 0.4)
            center = (object_size // 2 + i * object_size, object_size // 2)
            canvas = cv2.circle(canvas, center, radius, color, -1)
        elif shape == 1:
            pts = np.array([
                [object_size // 2 + i * object_size, object_size // 4],
                [object_size // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size * 3 // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)
        else:
            pts = np.array([
                [object_size // 4 + i * object_size, object_size // 4],
                [object_size // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size * 3 // 4],
                [object_size * 3 // 4 + i * object_size, object_size // 4]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))
            canvas = cv2.fillPoly(canvas, [pts], color)

    return canvas, shapes

def _gen_random_label():
    choices = ['circle', 'triangle', 'rectangle', 'red object', 'green object', 'blue object']
    return npr.choice(choices)


def _gen_random_question(shapes, arity: int):
    if arity == 1:
        label = _gen_random_label()
        label_primitive = label.split()[0]
        answer = any(shape['shape'] == label_primitive or shape['color'] == label_primitive for shape in shapes)
        return 'Is there a {}?'.format(label), f'exists(Object, lambda x: {label_primitive}(x))', answer
    else:
        label1 = _gen_random_label()
        label2 = _gen_random_label()
        label1_primitive = label1.split()[0]
        label2_primitive = label2.split()[0]
        relation = npr.choice(['left', 'right'])
        answer = False

        if relation == 'left':
            indices = [(0, 1), (1, 2), (0, 2)]
        else:
            indices = [(1, 0), (2, 1), (2, 0)]
        for i, j in indices:
            if shapes[i]['shape'] == label1_primitive or shapes[i]['color'] == label1_primitive:
                if shapes[j]['shape'] == label2_primitive or shapes[j]['color'] == label2_primitive:
                    answer = True
                    break

        return 'Is there a {} to the {} of a {}?'.format(label1, relation, label2), f'exists(Object, lambda x: exists(Object, lambda y: {label1_primitive}(x) and {relation}(x, y) and {label2_primitive}(y)))', answer


def gen_shapes3_dataset(dataset_size):
    images, objects, questions, programs, answers = list(), list(), list(), list(), list()
    for i in range(dataset_size):
        image, shapes = create_shapes3()
        images.append(image)
        objects.append(shapes)

        arity = npr.choice([1, 2])
        answer = npr.choice([True, False])
        for trials in range(10):
            question, logical_form, pred_answer = _gen_random_question(shapes, arity)
            if pred_answer == answer:
                break

        questions.append(question)
        programs.append(logical_form)
        answers.append(pred_answer)

    return dict(images=images, objects=objects, questions=questions, programs=programs, answers=answers)


class Shapes3DatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, dataset_size):
        self.data = gen_shapes3_dataset(dataset_size)

    def _get_metainfo(self, index):
        return {
            'question': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': bool(self.data['answers'][index]),
            'question_length': len(self.data['questions'][index].split())
        }

    def __getitem__(self, index):
        return {
            'image': _to_image(self.data['images'][index]),
            'question': self.data['questions'][index],
            'program': self.data['programs'][index],
            'answer': bool(self.data['answers'][index])
        }

    def __len__(self):
        return len(self.data['images'])


def _to_image(image):
    image = image.transpose(2, 0, 1) / 255.0
    image = image.astype(np.float32)
    image = (image - 0.5) * 2
    return torch.tensor(image)


class Shapes3DatasetFilterableView(FilterableDatasetView):
    def make_dataloader(self, batch_size: int, shuffle: bool, drop_last: bool, nr_workers: int) -> DataLoader:
        collate_guide = {
            'question': 'skip',
            'program': 'skip',
            'answer': 'skip'
        }
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )

    def filter_question_length(self, length: int) -> 'Shapes3DatasetFilterableView':
        def filt(meta):
            return meta['question_length'] <= length
        return self.filter(filt, f'filter-qlength[{length}]')


def Shapes3Dataset(dataset_size) -> Shapes3DatasetFilterableView:
    return Shapes3DatasetFilterableView(Shapes3DatasetUnwrapped(dataset_size))