import time
import sys, random
import os
import copy
import numpy as np
from math import sin, cos, radians, pi

import pymunk
from pymunk import Vec2d

import argparse
import time
import numpy as np
import cv2
import copy
import matplotlib
import socket
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os

import torch
from torch.autograd import Variable
import sys


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


def var_norm(x):
    return torch.sqrt((x ** 2).sum()).item()


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + \
                   (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    return x.detach().cpu().numpy()


def render_Cradle(des_dir, filename, lim, states, states_gt=None, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    boxcolor = (0.5, 0.5, 0.5)
    edgecolor = None

    num_frame = states.shape[0]
    num_obj = states.shape[1]

    for i in range(num_frame):
        if i % 5 != 0:
            continue

        fig, ax = plt.subplots(1)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        boxes = []
        circles = []
        circles_gt = []
        circle_colors = []

        num_balls = num_obj // 2

        for j in range(num_balls):
            s0 = states[i, j]
            s1 = states[i, j + num_balls]
            plt.plot([s0[0], s1[0]], [s0[1], s1[1]], 'r-', lw=1)

            if states_gt is not None:
                s0 = states_gt[i, j]
                s1 = states_gt[i, j + num_balls]
                plt.plot([s0[0], s1[0]], [s0[1], s1[1]], 'r-', lw=1, alpha=0.5)

        for j in range(num_balls):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=25)
            circles.append(circle)
            circle_colors.append('b')

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=25)
                circles_gt.append(circle)

        for j in range(num_balls, num_obj):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=5)
            circles.append(circle)
            circle_colors.append('r')

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=5)
                circles_gt.append(circle)

        pc = PatchCollection(circles, facecolor=circle_colors, edgecolor=edgecolor)
        ax.add_collection(pc)
        if states_gt is not None:
            pc = PatchCollection(circles_gt, facecolor=circle_colors, edgecolor=edgecolor, alpha=0.5)
            ax.add_collection(pc)
        ax.set_aspect('equal')

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if video:
            out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


def render_Rope(des_dir, filename, lim, attrs, states, actions=None, states_gt=None,
                count_down=True, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    time_step = states.shape[0]
    n_particle = states.shape[1]
    n_b = np.sum(attrs[:, 0] == 1) + 1

    for i in range(time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])

        if actions is not None:
            assert (actions[i, :-4] == 0).all()
            assert (actions[i, -2:] == 0).all()
            F = (actions[i, -4] + actions[i, -3]) / 2.
            normF = norm(F)
            Fx = F / normF * np.sqrt(np.sqrt(normF)) * 15.
            st = (states[i, -4, :2] + states[i, -3, :2]) / 2. + F / normF * 20.
            ax.arrow(st[0], st[1], Fx[0], Fx[1], fc='Orange', ec='Orange', width=6., head_width=30., head_length=30.)

        circles = []
        circles_gt = []
        circles_color = []

        for j in range(n_particle):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=attrs[j, 2])
            circles.append(circle)

            if states_gt is not None:
                circle = Circle((states_gt[i, j, 0], states_gt[i, j, 1]), radius=attrs[j, 2])
                circles_gt.append(circle)

            if j < n_b:
                circles_color.append('r')
            else:
                circles_color.append('b')

            if 0 < j < n_b:
                if states_gt is not None:
                    plt.plot([states_gt[i, j, 0], states_gt[i, j - 1, 0]],
                             [states_gt[i, j, 1], states_gt[i, j - 1, 1]], 'r-', lw=1, alpha=0.3)

                plt.plot([states[i, j, 0], states[i, j - 1, 0]],
                         [states[i, j, 1], states[i, j - 1, 1]], 'r-', lw=1, alpha=0.5)

        pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=1)
        ax.add_collection(pc)

        if states_gt is not None:
            pc = PatchCollection(circles_gt, facecolor=circles_color, linewidth=0, alpha=0.5)
            ax.add_collection(pc)

        ax.set_aspect('equal')

        if count_down:
            plt.text(0, 500, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        write_times = 25 if i == time_step - 1 else 1

        if video:
            for j in range(write_times):
                out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


def render_Box(des_dir, filename, lim, states, actions, vis, states_gt=None, vis_gt=None,
               count_down=True, video=False, image=False):

    color = ['r', 'b', 'g', 'k', 'y', 'm', 'c']

    if video:
        video_name = os.path.join(des_dir, filename) + '.avi'
        print('Render video ' + video_name)
        os.system('mkdir -p ' + des_dir)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(video_name, fourcc, 25, (640, 480))

    if image:
        image_name = os.path.join(des_dir, filename)
        print('Render image ' + image_name)
        os.system('mkdir -p ' + image_name)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    wallcolor = (0.5, 0.5, 0.5)
    edgecolor = None

    time_step = states.shape[0]
    n_particle = states.shape[1]

    for i in range(time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])

        boxes = []
        box_colors = []

        boxes_goal = []
        box_colors_goal = []

        radius = 15 - 0.5

        for j in range(n_particle):
            t = mpl.transforms.Affine2D().rotate_deg_around(
                states[i, j, 0], states[i, j, 1], np.rad2deg(states[i, j, 2]))

            box = Rectangle((states[i, j, 0] - radius, states[i, j, 1] - radius),
                            radius*2, radius*2, transform=t)
            boxes.append(box)

            if vis[i, j]:
                box_colors.append('tomato')
            else:
                box_colors.append('royalblue')

            if states_gt is not None and vis_gt is not None:
                t = mpl.transforms.Affine2D().rotate_deg_around(
                    states_gt[i, j, 0], states_gt[i, j, 1], np.rad2deg(states_gt[i, j, 2]))

                box = Rectangle(
                    (states_gt[i, j, 0] - radius, states_gt[i, j, 1] - radius),
                    radius*2, radius*2, transform=t)
                boxes_goal.append(box)

                if vis_gt[i, j]:
                    box_colors_goal.append('tomato')
                else:
                    box_colors_goal.append('royalblue')

        boxes.append(Rectangle((-600, -10), 1120, 20))
        box_colors.append(wallcolor)
        if actions is not None:
            boxes.append(Rectangle((actions[i, 0, 0] - 10, 10), 20, 300))
            box_colors.append(wallcolor)

        pc = PatchCollection(boxes, facecolor=box_colors, edgecolor='k', linewidth=0.3)
        ax.add_collection(pc)

        if states_gt is not None and vis_gt is not None:
            pc = PatchCollection(boxes_goal, facecolor=box_colors_goal, edgecolor='k', alpha=0.4)
            ax.add_collection(pc)

        ax.set_aspect('equal')

        if count_down:
            plt.text(-600, 300, 'CountDown: %d' % (time_step - i - 1), fontdict=font)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        write_times = 25 if i == time_step - 1 else 1

        if video:
            for j in range(write_times):
                out.write(frame)

        if image:
            cv2.imwrite(os.path.join(des_dir, filename, '%d.jpg' % i), frame)

        plt.close()

    if video:
        out.release()


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
from progressbar import ProgressBar

import torch


class CradleEngine(object):

    def __init__(self, dt=0.001):
        self.dt = dt
        self.radius = 25
        self.mass = 10

    def reset_scene(self, n_objects, theta, theta_right=None):

        self.space = pymunk.Space()
        self.space.gravity = (0.0, -4000.0)
        self.space.damping = 0.999

        st = -(n_objects - 1) * self.radius
        ed = (n_objects + 1) * self.radius

        bodies = []
        for x in range(int(st), int(ed), int(self.radius * 2)):
            mass = self.mass
            radius = self.radius
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            body = pymunk.Body(mass, moment)

            if x == st:
                # if it is the left-most ball, place it according to angle
                length = 250
                theta_rad = radians(theta)
                body.position = (x - length * sin(theta_rad), 125 - length * cos(theta_rad))
            else:
                body.position = (x, -125)

            # if specified the right position
            if x == ed - self.radius * 2 and (theta_right is not None):
                length = 250
                theta_rad = radians(theta_right)
                body.position = (x - length * sin(theta_rad), 125 - length * cos(theta_rad))

            #print(np.array(body.position))
            body.start_position = Vec2d(np.array(body.position))
            shape = pymunk.Circle(body, radius)
            shape.elasticity = 0.9999999
            self.space.add(body, shape)
            bodies.append(body)
            pj = pymunk.PinJoint(self.space.static_body, body, (x, 125), (0, 0))
            self.space.add(pj)

        return self.space

    def get_state(self):
        space = self.space
        idx = 0
        radius = 25
        y = 125

        n_objects = len(space.shapes)
        state = np.zeros((n_objects * 2, 4))
        for ball in space.shapes:
            state[idx, :2] = np.array([ball.body.position[0], ball.body.position[1]])
            state[idx, 2:] = np.array([ball.body.velocity[0], ball.body.velocity[1]])
            idx += 1

        st = -(n_objects - 1) * radius
        ed = (n_objects + 1) * radius
        for x in range(int(st), int(ed), int(radius * 2)):
            state[idx, :2] = np.array([x, y]); idx += 1

        return state

    def step(self):
        self.space.step(self.dt)
        return self.space


class RopeEngine(object):

    def __init__(self, dt, state_dim, action_dim, num_cylinder=2):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.space = None
        self.balls = None
        self.radius = 5
        self.actions = None
        self.cylinders = None
        self.num_cylinder = num_cylinder
        self.c_positions = None
        self.c_radius = None

        self.ticks_before_action = 50

    def init(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

    def add_balls(self, num=15, center=(300, 300), ckp=None):
        from pymunk import DampedSpring
        space = self.space

        radius = self.radius
        restlen = radius * 2

        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))

        balls = []

        # add fixed center mass
        if ckp is None:
            center += (np.random.rand(2) - 0.5) * 50
        else:
            center = ckp

        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = center
        shape = pymunk.Circle(body, radius, (0, 0))
        space.add(body, shape)
        balls.append(body)

        # add the rest of the masses
        for i in range(1, num):
            body = pymunk.Body(mass, inertia)
            body.position = center[0] + i * restlen, center[1]
            shape = pymunk.Circle(body, radius, (0, 0))
            space.add(body, shape)
            balls.append(body)

        for a, b in zip(balls[:-1], balls[1:]):
            spring = DampedSpring(a, b, (0, 0), (0, 0), rest_length=restlen, stiffness=100, damping=50)
            space.add(spring)

        for a, b in zip(balls[:-2], balls[2:]):
            spring = DampedSpring(a, b, (0, 0), (0, 0), rest_length=restlen * 2, stiffness=100, damping=50)
            space.add(spring)

        return balls, center

    def add_cylinders(self, num=1, ckp=None):
        from scipy.spatial.distance import pdist

        center_lim = [150, 450]
        radius_lim = [30, 60]

        if ckp is None:
            while True:
                '''sample center'''
                while True:
                    center = np.random.rand(num, 2) * (center_lim[1] - center_lim[0]) + center_lim[0]
                    D = pdist(center)
                    if len(D) == 0 or D.min() > radius_lim[0] * 2.1:
                        break

                '''sample radius'''
                radius = np.zeros(num) + radius_lim[0]

                for i in range(num):
                    upper_bound = np.sqrt(((center[i:i + 1, :] - center) ** 2).sum(1)) - radius
                    upper_bound[upper_bound < 0] = 300
                    upper_bound = upper_bound.min()
                    l, r = radius_lim[0], min(radius_lim[1], upper_bound)
                    radius[i] = np.random.rand() * (r - l) + l

                '''check collision with rope'''
                ok = True
                for C, R in zip(center, radius):
                    for ball in self.balls:
                        if np.sqrt(sum((C - np.array(ball.position)) ** 2)) - R - 10 < 0:
                            ok = False
                            break
                if ok:
                    break
        else:
            center, radius = ckp[0], ckp[1]

        self.cylinders = []
        for C, R in zip(center, radius):
            cylinder = pymunk.Body(body_type=pymunk.Body.STATIC)
            cylinder.position = C
            shape = pymunk.Circle(cylinder, R, (0, 0))
            self.space.add(cylinder, shape)
            self.cylinders.append(cylinder)

        self.c_positions = center
        self.c_radius = radius

        ckp = [center, radius]
        return ckp

    def reset_scene(self, n_particle, ticks_before_action=None, ckp=None):
        from random import randint

        self.init()
        if ckp is None:
            self.balls, balls_ckp = self.add_balls(num=n_particle)
        else:
            self.balls, balls_ckp = self.add_balls(num=n_particle, ckp=ckp[0])

        if ckp is None:
            cylinders_ckp = self.add_cylinders(num=self.num_cylinder)
        else:
            cylinders_ckp = self.add_cylinders(num=self.num_cylinder, ckp=ckp[1])

        self.actions = np.zeros((n_particle, 2))

        if ticks_before_action is not None:
            ticks = ticks_before_action
        else:
            ticks = self.ticks_before_action

        if ckp is None:
            act_ckp = []
            for i in range(ticks):
                act_t = np.random.normal(0, 20, size=(n_particle, self.action_dim))
                act_ckp.append(act_t)
                self.set_action(action=act_t)
                self.step()
        else:
            act_ckp = ckp[2]
            for i in range(ticks):
                act_t = act_ckp[i]
                self.set_action(action=act_t)
                self.step()

        ckp = [balls_ckp, cylinders_ckp, act_ckp]
        return ckp

    def get_state(self):
        n = len(self.balls)
        states = np.zeros((n, 4))
        # print(self.balls[0].position)
        for i in range(n):
            ball = self.balls[i]
            # print(angle, box.body.angle)
            states[i, :2] = np.array([ball.position[0], ball.position[1]])
            states[i, 2:] = np.array([ball.velocity[0], ball.velocity[1]])

        return states

    def set_action(self, action):
        self.actions = action

    def get_action(self):
        return self.actions

    def step(self):
        for impulse, ball in zip(self.actions, self.balls):
            ball.apply_impulse_at_local_point(impulse=impulse, point=(0, 0))
        self.space.step(self.dt)


class BoxEngine(object):

    def __init__(self, dt, state_dim, action_dim, num_particle=20):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.radius = 15
        self.mass = 1
        self.ticks_to_next_box = 10
        self.ticks_before_action = 50

    def init(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -900.0)
        self.pusher_vel = 0.

    def add_box(self, ckp=None):
        points = [(-self.radius, -self.radius),
                  (-self.radius, self.radius),
                  (self.radius, self.radius),
                  (self.radius, -self.radius)]
        inertia = pymunk.moment_for_poly(self.mass, points, (0, 0))
        body = pymunk.Body(self.mass, inertia)
        if ckp is None:
            x = random.randint(400, 450)
        else:
            x = ckp
        body.position = x, 200
        shape = pymunk.Poly(body, points)
        shape.friction = 0.6
        self.space.add(body, shape)
        return shape, x

    def add_floor(self):
        static_line = pymunk.Segment(self.space.static_body, (-600, 0), (520, 0), 10)
        static_line.friction = 0.6
        self.space.add(static_line)
        return static_line

    def add_pusher(self):
        body = pymunk.Body(1e7, pymunk.inf)
        body.position = 500, 10
        shape = pymunk.Segment(body, (0, 0), (0, 300), 10)
        shape.elasticity = 0.1
        shape.friction = 0.6
        self.space.add(body, shape)
        return body, shape

    def reset_scene(self, n_particle, ticks_before_action=None, ckp=None):
        self.init()
        self.floor = self.add_floor()
        self.pusher_body, self.pusher_shape = self.add_pusher()

        self.boxes = []
        self.n_particle = n_particle

        if ckp is None:
            ckp = []
            for i in range(n_particle):
                box, box_ckp = self.add_box()
                ckp.append(box_ckp)
                self.boxes.append(box)
                for j in range(self.ticks_to_next_box):
                    self.step()
        else:
            for i in range(n_particle):
                box, box_ckp = self.add_box(ckp[i])
                self.boxes.append(box)
                for j in range(self.ticks_to_next_box):
                    self.step()

        if ticks_before_action is not None:
            ticks = ticks_before_action
        else:
            ticks = self.ticks_before_action

        for i in range(ticks):
            self.step()

        return ckp

    def get_state(self):
        states = np.zeros((len(self.boxes), self.state_dim))
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            angle = box.body.angle % (2 * np.pi)
            # print(angle, box.body.angle)
            states[i, :3] = np.array([box.body.position[0], box.body.position[1], angle])
            states[i, 3:] = np.array([box.body.velocity[0], box.body.velocity[1], box.body.angular_velocity])
        return states

    def set_action(self, vel):
        self.pusher_vel = vel

    def get_action(self):
        action = np.zeros((1, self.action_dim))
        action[0, 0] = self.pusher_shape.body.position[0]
        action[0, 1] = self.pusher_shape.body.velocity[0]
        return action

    def get_vis(self, states):
        N = states.shape[0]
        buf = np.tile(states[:, :2], N).reshape((N, N, 2)) # N * N * 2
        buf -= states[:, :2]
        buf = np.logical_and(buf[:, :, 1] > 0, np.abs(buf[:, :, 0]) < self.radius * 1.2)
        buf = np.sum(buf.astype(np.int), axis=0)
        vis = buf == 0
        return vis

    def step(self):
        # print('set', self.pusher_vel)
        self.pusher_body.velocity = (self.pusher_vel, 0.)
        self.space.step(self.dt)


def test_gen_Cradle(info):

    thread_idx, n_balls, n_rollout, time_step, dt, video, image, data_dir = \
            info['thread_idx'], info['n_balls'], info['n_rollout'], info['time_step'], info['dt'], \
            info['video'], info['image'], info['data_dir']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    lim = 400
    attr_dim = 2
    state_dim = 4
    relation_dim = 4

    engine = CradleEngine(dt)

    n_objects = n_balls * 2
    states = np.zeros((n_rollout, time_step, n_objects, state_dim))

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        theta = rand_float(0, 90)
        engine.reset_scene(n_balls, theta)

        for j in range(time_step):
            states[i, j] = engine.get_state()
            if j > 0:
                states[i, j, :, 2:] = (states[i, j, :, :2] - states[i, j - 1, :, :2]) / dt
            engine.step()

        if video or image:
            render_Cradle(data_dir, 'test_cradle_%d' % i, lim, states[i], video=video, image=image)


def test_gen_Rope(info):
    thread_idx, data_dir = info['thread_idx'], info['data_dir']
    n_rollout, n_particle, time_step = info['n_rollout'], info['n_particle'], info['time_step']
    dt, video, image = info['dt'], info['video'], info['image']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    state_dim = 4  # x, y, xdot, ydot
    action_dim = 2  # xdotdot, ydotdot
    act_scale = 15

    engine = RopeEngine(dt, state_dim, action_dim)

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        # os.system('mkdir -p ' + rollout_dir)

        engine.reset_scene(n_particle)

        all_p = []

        act_t = np.zeros((n_particle, action_dim))
        for j in range(time_step):

            f = np.zeros(action_dim)
            for k in range(n_particle):
                f += (np.random.rand(action_dim) * 2 - 1) * act_scale
                act_t[k] = f

            engine.set_action(action=act_t)

            states = engine.get_state()
            actions = engine.get_action()

            pos = states[:, :2]
            vec = states[:, 2:]

            '''reset velocity'''
            if len(all_p) > 0:
                vec = (pos - all_p[-1]) / dt

            all_p.append(pos)

            n_b = len(states)
            n_c = len(engine.cylinders)

            '''attrs: 0 => moving; 1 => fixed'''
            attrs = np.zeros((n_b + n_c, 3))
            attrs[0, 1] = 1
            attrs[1:n_b, 0] = 1
            attrs[n_b:, 1] = 1
            attrs[:n_b, 2] = 5
            attrs[n_b:, 2] = engine.c_radius

            positions = np.concatenate([pos, engine.c_positions], axis=0)
            velocities = np.concatenate([vec, np.zeros((n_c, 2))], axis=0)
            radius = np.concatenate([np.array([5] * n_b), engine.c_radius], axis=0)

            states = np.concatenate([positions, velocities], 1)

            # data = [attrs, positions, velocities, actions]
            # store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        if video or image:
            lim = [0, 600, 0, 600]

            ball_p = np.array(all_p)
            cylinder_p = np.array([engine.c_positions] * len(ball_p))

            states = np.concatenate([ball_p, cylinder_p], axis=1)
            render_Rope(data_dir, 'test_Rope_%d' % i, lim, attrs, states, video=video, image=image)


def test_gen_Box(info):

    thread_idx, data_dir = info['thread_idx'], info['data_dir']
    n_rollout, n_particle, time_step = info['n_rollout'], info['n_particle'], info['time_step']
    dt, video, image = info['dt'], info['video'], info['image']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2**32)

    state_dim = 6       # x, y, angle, xdot, ydot, angledot
    action_dim = 2      # x, xdot

    stats = [init_stat(6), init_stat(2)]

    engine = BoxEngine(dt, state_dim, action_dim)

    bar = ProgressBar()
    for i in bar(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        # os.system('mkdir -p ' + rollout_dir)

        engine.reset_scene(n_particle)

        states = np.zeros((time_step, n_particle, state_dim), dtype=np.float32)
        actions = np.zeros((time_step, 1, action_dim), dtype=np.float32)
        vis = np.zeros((time_step, n_particle), dtype=np.bool)

        for j in range(time_step):
            engine.set_action(rand_float(-600., 100.))

            states[j] = engine.get_state()
            actions[j] = engine.get_action()
            vis[j] = engine.get_vis(states[j])

            if j > 0:
                actions[j, :, 1] = (actions[j, :, 0] - actions[j - 1, :, 0]) / dt

            # data = [states[j], actions[j], vis[j]]
            # store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            engine.step()

        # datas = [states.astype(np.float64), actions.astype(np.float64)]

        '''
        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)
        '''

        if video:
            lim = [-600, 600, -15, 400]

            if video or image:
                render_Box(data_dir, 'test_Box_%d' % i, lim, states, actions, vis, video=video, image=image)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="Cradle")
    args = parser.parse_args()

    if args.env == 'Cradle':
        test_gen_Cradle({
            'thread_idx': 0,
            'data_dir': 'test/test_data_Cradle',
            'n_balls': 5,
            'n_rollout': 5,
            'time_step': 1000,
            'dt': 1./1000.,
            'video': True,
            'image': True
        })

    elif args.env == 'Rope':
        test_gen_Rope({
            'thread_idx': 0,
            'data_dir': 'test/test_data_Rope',
            'n_particle': 15,
            'n_rollout': 5,
            'time_step': 100,
            'dt': 1./50.,
            'video': True,
            'image': True
        })

    elif args.env == 'Box':
        test_gen_Box({
            'thread_idx': 0,
            'data_dir': 'test/test_data_Box',
            'n_particle': 20,
            'n_rollout': 5,
            'time_step': 100,
            'dt': 1./50.,
            'video': True,
            'image': True
        })

    else:
        raise AssertionError("Unsupported env")