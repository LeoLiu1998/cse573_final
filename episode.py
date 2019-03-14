""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS, DONE_TOMATO, DONE_BOWL
from environment import Environment
from utils.net_util import gpuify
import numpy as np
from math import sqrt


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects
        self.actions_list = [{'action': a} for a in BASIC_ACTIONS]
        self.actions_taken = []
        self.done_each_obj = [0, 0]  # store agents' judgements
        self.successes = [0, 0]
        # self.seen_objects = [0 for _ in range(len(self.objects))]
        self.success = False
        self.distances = [float('inf'), float('inf')]
        self.args = args

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed=False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    @staticmethod
    def cal_distance(pos1, pos2):
        dx = pos1['x'] - pos2['x']
        dy = pos1['y'] - pos2['y']
        dz = pos1['z'] - pos2['z']
        return sqrt(dx**2 + dy**2 + dz**2)

    def judge(self, action):
        """ Judge the last event. """
        # TODO: change for two objects
        # immediate reward
        reward = STEP_PENALTY 
        all_done = False
        action_was_successful = self.environment.last_action_success
        if self.args.improve:
            objects = self._env.last_event.metadata['objects']
            agent_pos = self._env.last_event.metadata['agent']['position']

            visible_objects = [o['objectType'] for o in objects if o['visible']]
            for i in range(len(self.distances)):
                target = self.target[i]
                if target in visible_objects:
                    object_meta = [o for o in objects if o['objectType'] == target]
                    assert len(object_meta) == 1
                    object_meta = object_meta[0]
                    pos = object_meta['position']
                    distance2agent = self.cal_distance(pos, agent_pos)
                    if distance2agent < self.distances[i] and self.done_each_obj[i] != 1:
                        # if we are getting closer to the object \and
                        # the object is not "done"(consider by the agent) yet.
                        reward = 0
                        self.distances[i] = distance2agent

        if action['action'] in [PICKUP_OBJECT, COOK]:

            done_id = [DONE_TOMATO, DONE_BOWL].index(action['action'])
            if not self.args.many_dones:
                if self.done_each_obj[done_id] != 1:
                    self.done_each_obj[done_id] = 1

                    objects = self._env.last_event.metadata['objects']
                    visible_objects = [o['objectType'] for o in objects if o['visible']]
                    if self.target[done_id] in visible_objects:
                        reward += SUCCESS_REWARD
                        self.successes[done_id] = 1
                        self.success = all(self.successes)
                    if self.args.bonus:
                        if self.success:
                            reward += GOAL_SUCCESS_REWARD
            else:
                if self.successes[done_id] != 1:

                    objects = self._env.last_event.metadata['objects']
                    visible_objects = [o['objectType'] for o in objects if o['visible']]
                    if self.target[done_id] in visible_objects:
                        reward += SUCCESS_REWARD
                        self.successes[done_id] = 1
                        self.success = all(self.successes)
                    if self.args.bonus:
                        if self.success:
                            reward += GOAL_SUCCESS_REWARD

        if self.args.many_dones:
            all_done = self.success
        else:
            all_done = all(self.done_each_obj)
        return reward, all_done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.BowlTomato
        self.target = ['Tomato', "Bowl"]
        self.success = False
        self.done_each_obj = [0, 0]
        self.successes = [0, 0]
        self.distances = [float('inf'), float('inf')]
        self.cur_scenecur_scene = scene
        self.actions_taken = []
        
        return True
