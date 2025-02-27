#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import time
import csv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

# import cv2
import pickle 

import carla

from carla import ColorConverter as cc

import argparse
import collections
from collections import Counter
import datetime
import logging
import math
import random
import string
import re
import weakref
#random.seed(0)
#import numpy.random as random

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.basic_agent_sa import BasicAgentSA
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

# ==============================================================================
# -- Global variables ----------------------------------------------------------
# ==============================================================================

global origin
origin = carla.Location(x = -664, y = 89, z=0.9)#carla.Location(x = -780, y = 560, z=-7)
global destination
end_agent_destination = carla.Location(x=-600, y=274, z=0)#carla.Location(x=-781, y=560, z=-7)
destination = carla.Location(x = -665, y = 89, z=0.6)#carla.Location(x=-781, y=560, z=-7) -747, 289, -5.4
global USER 
USER = "user"
global MAX_SPEED 
MAX_SPEED = 14000
TRACK_FILE = './track.csv'
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# Define the time limit T (in seconds)
TIME_LIMIT = 180  # 3 minutes


def calculate_reward(simulation_time):
    if simulation_time > 0 and simulation_time < TIME_LIMIT:
        return TIME_LIMIT - simulation_time
    if simulation_time > TIME_LIMIT:
        return -TIME_LIMIT
    return 0


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter, mode, trial, username):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.front_car = None
        
        self.back_car = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        # self._actor_filter = actor_filter
        self._actor_filter = 'vehicle.toyota.prius'
        self._mode = mode
        self._trial = trial
        self.data_buffer = None
        self.current_episode = None
        self.episode_id = None
        self.obs_path = None
        self.unique_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        # self.data_log = 'data/driving_data_'+str(int(time.time()))+'.pkl'
        self.data_log = 'data/'+ username + "_" + "trial"+ str(self._trial) + "_" + self._mode.lower() +"_"+self.unique_id +'.pkl'
        self.agent = None
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def draw_street_barrier(self, location, yaw=90):
        street_barrier_blueprint = self.world.get_blueprint_library().find('static.prop.streetbarrier')
        # Get the blueprint for the street barrier
        spawn_point_street_barrier = carla.Transform(carla.Location(x=location.x, y=location.y, z=location.z), carla.Rotation(yaw=yaw))
        self.street_barrier = self.world.try_spawn_actor(street_barrier_blueprint, spawn_point_street_barrier)
        
    def draw_boundary_marker(self, location):
        self.world.debug.draw_point(location, size=0.07, color=carla.Color(1, 0, 0), life_time=0)

    def draw_optimal_path(self, filename):
        # Open the .pkl file in binary read mode
        with open(filename 'rb') as file:
            data = pickle.load(file)

        if isinstance(data, dict):
            data_dict = data

        data = data_dict[list(data_dict.keys())[0]]
        for data_point in data["state"]:
            location = data_point
            loc = carla.Location(x=location['x'], y=location['y'], z=location['z'])  # Raise z for better visibility
            self.world.debug.draw_point(loc, size=0.05, color=carla.Color(0, 255, 255), life_time=0)


    def restart(self):
        
        if self.data_buffer is None:
            self.data_buffer = {}

        if self.current_episode is not None:
            self.data_buffer[self.episode_id] = self.current_episode
            print('Restart! Saving recorded data to ' + self.data_log + ' ...')
            with open(self.data_log, 'wb') as f:
                pickle.dump(self.data_buffer, f)
        
        self.episode_id = int(time.time())
        self.current_episode = {'initial_state': {}, 'state':[], 'rotation':[], 'velocity':[], 'control' : {"user" : [], "agent" : [], "mixed" : []}, 'time': [], "lap_time" : [], 'reward': None, 'num_invasions': None}

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # Set this to be any car in the CARLA car library
        # controlled_car = 'vehicle.mercedes.coupe_2020'
        blueprint = self.world.get_blueprint_library().find(self._actor_filter)
        #blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')

        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '135, 42, 150') # set a static color


        spawn_point = carla.Transform(origin, carla.Rotation(yaw=270))


        self.current_episode['initial_state']['ego_car'] = (origin.x,origin.y)

        if self.player is not None: self.destroy()
        
        while self.player is None:            
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            # self.player.set_autopilot(True)
        self.agent = BasicAgent(self.player, MAX_SPEED)
        self.agent.follow_speed_limits(False)
        self.agent.set_destination(end_location=end_agent_destination, start_location=origin)

 
        ########## PATH ##########
        if self._mode == "WeakSA"or self._mode == "StrongSA" or self._mode == "Steer" or self._mode == "Throttle" or self._mode == "Brake" or self._mode == "Practice":
            self.draw_optimal_path("expert_path.pkl")
            
        
        ##### draw street barriers #####
        
        BARRIER_COORDINATES = [
            # first
	    [-628.001,-152.25, 7.641, 90],
	    [-628.001,-150.25, 7.641, 90],
	    # second
	    [-670.2, -291.694, 9.91, 90],
	    [-670.2, -289.694, 9.91, 90],
	    [-670.2, -287.694, 9.91, 90],
	    # third
	    [-611.93, 567.09, -11.368, 0],
	    [-608.93, 567.09, -11.368, 0],
	    # fourth
	    [-629.518, 381.885, -3.888, 0], 
	    [-629.518, 383.885, -3.888, 0],
	    # fifth
	    [-631.123, 372.488, -4.073, 90],
	    [-631.123, 375.488, -4.073, 90]]
    
        for bar in BARRIER_COORDINATES:

            street_barrier_location = carla.Location(x=bar[0], y=bar[1], z = bar[2])
            self.draw_street_barrier(street_barrier_location, yaw=bar[3])
        
        file_path = TRACK_FILE
        
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])


    def draw_finish_line(self, location):
        # Draw a thick, long finish line at the specified location
        line_width = 10.0  # Width of the finish line (adjust as needed)
        line_thickness = 0.2  # Thickness of the finish line
        color = carla.Color(255, 0, 0)  # Red color

        # Define the start and end points of the finish line
        start_point = carla.Location(location.x-5 , location.y, location.z)
        end_point = carla.Location(location.x+5, location.y, location.z)

        # Draw the rectangle
        
        self.world.debug.draw_line(start_point, end_point, thickness=line_thickness, color=color, life_time=0)

        # Add additional lines to make it appear thicker
        for i in range(1, 6):
            offset = i * line_thickness
            self.world.debug.draw_line(carla.Location(start_point.x, start_point.y - offset, start_point.z),
                                       carla.Location(end_point.x, end_point.y - offset, end_point.z),
                                       thickness=line_thickness, color=color, life_time=0)
            self.world.debug.draw_line(carla.Location(start_point.x, start_point.y + offset, start_point.z),
                                       carla.Location(end_point.x, end_point.y + offset, end_point.z),
                                       thickness=line_thickness, color=color, life_time=0)

    def tick(self, clock):
        
        self.hud.tick(self, clock)
        #print('img path:', self.obs_path)

        # Get all of the values in the current state
        # time_stamp = str(round(self.hud.simulation_time, 4)) + ","
        time_stamp = str(round(self.hud.simulation_time, 4))
        
        t = self.player.get_transform()
        v = self.player.get_velocity()
        c = self.player.get_control()

        curr_state = {"x": t.location.x, "y" : t.location.y, "z" : t.location.z}
        curr_rotation = [t.rotation.pitch, t.rotation.yaw, t.rotation.roll]
        curr_velocity = [v.x, v.y, v.z]


        self.current_episode['time'].append(time_stamp)
        self.current_episode['state'].append(curr_state)
        self.current_episode['rotation'].append(curr_rotation)
        self.current_episode['velocity'].append(curr_velocity)

        # Draw the finish line
        finish_line_location = carla.Location(x=destination.x, y=destination.y, z=destination.z)
        self.draw_finish_line(finish_line_location)

    def render(self, display):

        self.obs_path = self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        if self.current_episode is not None:
            self.data_buffer[self.episode_id] = self.current_episode
            print('Destroy! Saving recorded data to ' + self.data_log + ' ...')
            with open(self.data_log, 'wb') as f:
                pickle.dump(self.data_buffer, f)
        
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
            self.player = None
        if self.front_car is not None:
            self.front_car.destroy()
            self.front_car = None
        if self.back_car is not None:
            self.back_car.destroy()
            self.back_car = None

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        #world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                    #pygame.joystick.init()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    #world.camera_manager.toggle_camera()
                    pass
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                    world.camera_manager.toggle_camera()
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 2.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.mode = ""
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        
    def set_mode(self, mstr):
        self.mode = mstr

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        world.current_episode['num_invasions'] = world.lane_invasion_sensor.lane_invasion_count


        # Check if the car has reached the specified destination, verify location
        if (t.location.y > 92 and t.location.y <= 100) and (t.location.x >= -670 and t.location.x <= -660):
            print("Car has REACHED the specified DESTINATION. Calculating reward...")
            world.current_episode['lap_time'] = [self.simulation_time, self.simulation_time - float(world.current_episode['time'][0])]
            reward = calculate_reward(world.current_episode['lap_time'][1])
            world.current_episode['reward'] = reward
            world.destroy()
            return
        
        # Check if the time limit has been exceeded
        try:
            if (self.mode in ["Baseline", "StrongSA", "WeakSA"] and (self.simulation_time - float(world.current_episode['time'][0]) >= TIME_LIMIT)):
                print("TIME LIMIT exceeded. Calculating reward...")
                world.current_episode['lap_time'] = [self.simulation_time, self.simulation_time - float(world.current_episode['time'][0])]
                reward = calculate_reward(world.current_episode['lap_time'][1])
                world.current_episode['reward'] = reward
                world.destroy()
                return
        except:
            pass

        self._info_text = ['Speed: % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))]

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]


    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if(self.mode == "Steer"):
            skill_text = "Practice smooth steering!!"
            font = pygame.font.SysFont("Roboto", 100)
            skill_text_surface = font.render(skill_text, True, (25, 25, 25))
            display_width, display_height = display.get_size()
            display.blit(skill_text_surface, (display_width / 2 - skill_text_surface.get_width() / 2, 10))
        elif(self.mode == "Throttle"):
            # practice skill, TODO highlight lane color!
            skill_text = "Practice stable throttle!"
            font = pygame.font.SysFont("Roboto", 100)
            skill_text_surface = font.render(skill_text, True, (25, 25, 25))
            display_width, display_height = display.get_size()
            display.blit(skill_text_surface, (display_width / 2 - skill_text_surface.get_width() / 2, 10))
        elif(self.mode == "Brake"):
            # practice skill, TODO highlight lane color!
            skill_text = "Practice precise braking!"
            font = pygame.font.SysFont("Roboto", 100)
            skill_text_surface = font.render(skill_text, True, (25, 25, 25))
            display_width, display_height = display.get_size()
            display.blit(skill_text_surface, (display_width / 2 - skill_text_surface.get_width() / 2, 10))
        
        # hud
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.lane_invasion_count = 0
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_invasion_count += 1
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = True # record images
        self.img_path = None

        self._camera_transforms = [
            carla.Transform(carla.Location(x=0,z=1.2), carla.Rotation(0,0,0)), # ego view
            # carla.Transform(carla.Location(x=-8,z=3.2), carla.Rotation(0,0,0)), # ego view
            carla.Transform(carla.Location(x=-1.6, z=1.7), carla.Rotation(yaw=-180))] # reverse

        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
        return self.img_path

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if self.transform_index == 1:
                array = np.fliplr(array)
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)) # 720 1280
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            # img_obs = cv2.resize(array, (image.width//4,image.height//4))
            # img_obs = cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB)
            # self.img_path = 'data/img_obs/%08d.jpg' % image.frame
            # cv2.imwrite(self.img_path, img_obs)
            #image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================



def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def game_loop(args):
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((args.width, args.height),pygame.HWSURFACE | pygame.DOUBLEBUF)
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2000.0)

         # # Load the opendrive map
        vertex_distance = 2.0  # in meters
        max_road_length = 50.0 # in meters  10
        wall_height = 0.5     # in meters  1.0
        extra_width = 0.6      # in meters
        global waypoint_grid
        waypoint_grid = 5
        # global waypoint_search_grid
        # waypoint_search_grid = 10
        throttle_value = 0.9
        initialized_point = 0
        global section
        section = 500
        par_indices = []

        print("creating sim world")
        sim_world = client.load_world("ThunderHill")
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)
        sim_world = client.reload_world()
        print("loaded")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        hud.set_mode(args.mode)
        world = World(sim_world, hud, args.filter, args.mode, args.trial, args.username)
        controller = DualControl(world, args.autopilot)
        print("CHECK ALIVE")
        print(world.player.is_alive)
        

        # Set the agent destination
        world.agent.set_destination(end_location=end_agent_destination, start_location=origin)

        clock = pygame.time.Clock()
        
        #set physics control
        #print("SETTING PHYSICS CONTROL")
        vehicle_obj = random.choice(world.world.get_actors().filter('vehicle.*'))
        
        tire_front = carla.WheelPhysicsControl(tire_friction=3.05, damping_rate=0.250000, max_steer_angle=69.999992, radius=37.000000,
        max_brake_torque=3000, max_handbrake_torque=0.000000, lat_stiff_max_load=3.000000, lat_stiff_value=20.000000, long_stiff_value=3000.000000)
        
        tire_back = carla.WheelPhysicsControl(tire_friction=3.05, damping_rate=0.250000, max_steer_angle=0, radius=37.000000,
        max_brake_torque=3000, max_handbrake_torque=3000, lat_stiff_max_load=3.000000, lat_stiff_value=20.000000, long_stiff_value=3000.000000)
        
        
        wheels = [tire_front, tire_front, tire_back, tire_back]
        
        physics_control = vehicle_obj.get_physics_control()
        physics_control.torque_curve = [carla.Vector2D(0, 1000), carla.Vector2D(1890.760742, 900), carla.Vector2D(5729.577637, 700)]
        physics_control.max_rpm = 8750
        physics_control.moi= 1.0
        physics_control.damping_rate_full_throttle= 0.150000
        physics_control.damping_rate_zero_throttle_clutch_engaged= 2.000000
        physics_control.damping_rate_zero_throttle_clutch_disengaged= 0.350000
        physics_control.use_gear_autobox= True
        physics_control.gear_switch_time= 0.000000
        physics_control.clutch_strength= 10.000000
        physics_control.final_ratio= 4.170000
        physics_control.mass= 1775.000000
        physics_control.drag_coefficient=0.300000
        physics_control.center_of_mass=carla.Location(x=0.500000, y=0.000000, z=-0.300000)
        physics_control.steering_curve = [carla.Vector2D(0, 1.0), carla.Vector2D(20, 0.9), carla.Vector2D(60, 0.8), carla.Vector2D(120, 0.7)]
        physics_control.wheels = wheels
        vehicle_obj.apply_physics_control(physics_control)
        #print(physics_control)
        #print("DONE PHYSICS CONTROL")

        ALL_STATES = []
        ALL_ROTATIONS = []
        ALL_VELOCITIES = []
        #nn_files = ["3HHUU", "4RPJJ", "6SNXW", "MWIK9", "TG8N5"]
        #nn_files = ["4RPJJ" , "MWIK9"]
        # nn_files = ["3HHUU", "6SNXW", "TG8N5"]
        nn_files = ["MWIK9"]
        for file in nn_files:
            fn = 'nn_data/jackexpert2_baseline_1_'+ file + '.pkl'
            with open(fn, 'rb') as file:
                data = pickle.load(file)

            if isinstance(data, dict):
                data_dict = data
            episode_key = list(data_dict.keys())[0]
            #print(data_dict[episode_key].keys())
            states = data_dict[episode_key]["state"]
            rotations = data_dict[episode_key]["rotation"]
            ALL_STATES.append(states)
            ALL_VELOCITIES.append(data_dict[episode_key]["velocity"])
            ALL_ROTATIONS.append(rotations)
            
        force_user_control = False
        force_agent_control = False
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            ## user's control
            user_control = world.player.get_control()
            ## agent's control
            agent_control = None

            ## mixed control
            mixed_control = carla.VehicleControl()
            mixed_control.throttle = user_control.throttle
            mixed_control.steer = user_control.steer
            mixed_control.brake = user_control.brake
            mixed_control.hand_brake = user_control.hand_brake
            mixed_control.reverse = user_control.reverse
            mixed_control.manual_gear_shift = user_control.manual_gear_shift
            mixed_control.gear = user_control.gear

            if args.mode not in ["Baseline", "Plan"]: 
                #print(world.agent.vehicle.is_alive)
                #print(print(world.player.is_alive))
                ## agent's control
                agent_control = world.agent.run_step()
                agent_control.manual_gear_shift = False
                #print(agent_control)
                if force_user_control and args.mode in ["StrongSA", "WeakSA", "Practice"]:
                    alpha=0.0
                    mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                    mixed_control.brake = (alpha*agent_control.brake)+((1-alpha)*user_control.brake)
                    mixed_control.throttle = (alpha*agent_control.throttle)+((1-alpha)*user_control.throttle)
                elif args.mode == "Throttle":
                    alpha=0.05
                    if force_user_control:
                        mixed_control.steer = user_control.steer
                        mixed_control.brake = user_control.brake
                        mixed_control.throttle = (alpha*agent_control.throttle)+((1-alpha)*user_control.throttle)
                    else:
                        alpha=0.8
                        mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                        mixed_control.brake = (alpha*agent_control.brake)+((1-alpha)*user_control.brake)
                        mixed_control.throttle = user_control.throttle
                elif args.mode == "Brake":
                    alpha=0.05
                    if force_user_control:
                        mixed_control.steer = user_control.steer
                        mixed_control.brake = (alpha*agent_control.brake)+((1-alpha)*user_control.brake)
                        mixed_control.throttle = user_control.throttle
                    else:
                        alpha=1
                        mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                        mixed_control.brake = user_control.brake
                        mixed_control.throttle = (alpha*agent_control.throttle)+((1-alpha)*user_control.throttle)                      
                elif args.mode == "Steer":
                    alpha=0.05
                    if force_user_control:
                        mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                        mixed_control.brake = user_control.brake
                        mixed_control.throttle = user_control.throttle
                    else:
                        mixed_control.steer = user_control.steer
                        mixed_control.brake = agent_control.brake
                        mixed_control.throttle = agent_control.throttle
                elif args.mode == "StrongSA":
                    alpha=0.8
                    mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                    mixed_control.brake = (alpha*agent_control.brake)+((1-alpha)*user_control.brake)
                    mixed_control.throttle = (alpha*agent_control.throttle)+((1-alpha)*user_control.throttle)
                elif args.mode == "WeakSA":
                    alpha=0.05
                    mixed_control.steer = (alpha*agent_control.steer)+((1-alpha)*user_control.steer)
                    mixed_control.brake = (alpha*agent_control.brake)+((1-alpha)*user_control.brake)
                    mixed_control.throttle = (alpha*agent_control.throttle)+((1-alpha)*user_control.throttle)
                world.player.apply_control(mixed_control)
                
            world.current_episode['control']['user'].append({"throttle" : user_control.throttle, "steer" : user_control.steer, "brake" : user_control.brake, 
            "hand_brake" : user_control.hand_brake, "reverse" : user_control.reverse, "manual_gear_shift" : user_control.manual_gear_shift, "gear" : user_control.gear})
            
            if agent_control is None: 
                world.current_episode['control']['agent'].append(agent_control)
            else:
                world.current_episode['control']['agent'].append({"throttle" : agent_control.throttle, "steer" : agent_control.steer, "brake" : agent_control.brake,
                "hand_brake" : agent_control.hand_brake, "reverse" : agent_control.reverse, "manual_gear_shift" : agent_control.manual_gear_shift, "gear" : agent_control.gear})

            world.current_episode['control']['mixed'].append({"throttle" : mixed_control.throttle, "steer" : mixed_control.steer, "brake" : mixed_control.brake,
            "hand_brake" : mixed_control.hand_brake, "reverse" : mixed_control.reverse, "manual_gear_shift" : mixed_control.manual_gear_shift, "gear" : mixed_control.gear})
        
            if args.mode not in ["Baseline", "Plan"]: 
                nn_all = []
                t = world.player.get_transform()
                current_x = t.location.x
                current_y = t.location.y
                for s in range(len(ALL_STATES)):
                    states = ALL_STATES[s]
                    rotations = ALL_ROTATIONS[s]
                    current_index = None
                    min_distance = float('inf')
                    for idx, state in enumerate(states):
                        distance = calculate_distance(current_x, current_y, state['x'], state['y'])
                        if distance < min_distance:
                            min_distance = distance
                            current_index = idx
                    current_yaw = rotations[current_index][1] 
                    yaw_rad = math.radians(current_yaw)
                    heading_x = math.cos(yaw_rad)
                    heading_y = math.sin(yaw_rad)
                    points_in_front = []
                    # Loop over states ahead in time
                    for idx in range(current_index + 1, len(states)):
                        state = states[idx]
                        dx = state['x'] - current_x
                        dy = state['y'] - current_y
                        # Compute the dot product
                        dot_product = heading_x * dx + heading_y * dy
                        if dot_product > 0:
                            # Point is in front of the car
                            distance = calculate_distance(current_x, current_y, state['x'], state['y'])
                            state_info = {
                                'x': state['x'],
                                'y': state['y'],
                                'distance': distance,
                                'index': idx,
                                'par_index': s
                            }
                            points_in_front.append(state_info)
                    sorted_states = sorted(points_in_front, key=lambda loc: loc['distance'])
                    nn_all.append(sorted_states[0])
            
                sorted_nn_all = sorted(nn_all, key=lambda item: item['distance'])
                threshold = 1
                for nn in sorted_nn_all:
                    if nn["distance"] < threshold:
                        interfere = False
                future_steps = 500#200
                candidate_nn = sorted_nn_all[0]
                if(candidate_nn["distance"] > 20):
                    force_user_control = True
                    force_agent_control = True
                else:
                    force_user_control = False
                    force_agent_control = False
                par_indices.append(sorted_nn_all[0]['par_index'])
                candidate_state = ALL_STATES[candidate_nn['par_index']][candidate_nn['index']]
                if future_steps + candidate_nn['index'] < len(ALL_STATES[candidate_nn['par_index']]):
                    candidate_state = ALL_STATES[candidate_nn['par_index']][candidate_nn['index'] + future_steps]
                    candidate_velocity = np.linalg.norm(ALL_VELOCITIES[candidate_nn['par_index']][candidate_nn['index'] + future_steps])
                    temp_destination = carla.Location(x = candidate_state['x'], y = candidate_state['y'], z=candidate_state['z'])
                    temp_speed= candidate_velocity*3.6 
                else:
                    temp_destination = end_agent_destination
                world.agent.set_destination(end_location=temp_destination)
                world.agent.set_target_speed(temp_speed)
    finally:
        if world is not None:
            print("Finally!")
            world.destroy()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Basic")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-m', '--mode', type=str,
        choices=["Baseline", "Throttle", "Brake", "Steer", "Plan", "StrongSA", "WeakSA", "Practice"],
        help="Select which mode of shared autonomy to run",
        default="Baseline")
    argparser.add_argument(
        '-usr', '--username', type=str,
        required=True,
        help="username")
    argparser.add_argument(
        '-t', '--trial',
        required=True,
        type=int,
        help='user study trial')
    argparser.add_argument(
        '-l', '--lap',
        metavar='L',
        default=1,
        type=int,
        help='lap number')
    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()