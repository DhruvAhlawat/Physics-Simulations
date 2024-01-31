import pygame
import sys
import os
import numpy as np;
from pygame.locals import *
import time
import pickle
import random
# import numba 
# from numba import jit
pygame.mixer.pre_init(44100, -16, 2, 512) # setup mixer to avoid sound lag
pygame.init() #initializes pygame

## USAGE: python3 multibody.py [total_particles]
## ESC to quit, P to pause, R to reset.
physics_multiplier = 2; ## NOTE: If lagging occurs, reduce this parameter to 1. or otherwise reduce framerate below its current value. 
## NOTE: this is basically the number of physics steps per frame. on increasing this value, we get more accurate physics.

framerate = 64; #sets framerate to 64 FPS. A power of 2 ensures 1/framerate is exact.
gravity_acceleration = -1150; #this is the acceleration due to gravity in units per second squared.
elasticity = 0.8#1 means perfectly elastic, 0 means perfectly inelastic.

clock = pygame.time.Clock()
screen_width = 720; 
screen_height = 720;
screen = pygame.display.set_mode((screen_width, screen_height)) #creates screen
pygame.mouse.set_visible(0);
# bg = pygame.image.load("./images/bubble2_large.jpg") #loads background image
fixed_delta_seconds = 1/framerate;
total_particles = 30;
if(len(sys.argv) > 1):
    total_particles = int(sys.argv[1]);
#FORCE VARIABLES TO CONSIDER:

#BOUNDING BOX DETAILS:
#the bounding box is a rectangle that is aligned with the x and y axes.
lower_boundary = 0;
upper_boundary = screen_height;
left_boundary = 0;
right_boundary = screen_width;

def boundary_collision_check(obj):
    collided = False;
    if(obj.pos[1] - obj.radius < lower_boundary):
        obj.pos[1] = lower_boundary + obj.radius;
        obj.velocity[1] = - elasticity * obj.velocity[1];
        collided = True;

    if(obj.pos[1] + obj.radius > upper_boundary):
        obj.pos[1] = upper_boundary - obj.radius;
        obj.velocity[1] = - elasticity * obj.velocity[1];
        collided = True;

    if(obj.pos[0] - obj.radius < left_boundary):
        obj.pos[0] = left_boundary + obj.radius;
        obj.velocity[0] = - elasticity * obj.velocity[0];
        collided = True;

    if(obj.pos[0] + obj.radius > right_boundary):
        obj.pos[0] = right_boundary - obj.radius;
        obj.velocity[0] = - elasticity * obj.velocity[0];
        collided = True;
    return collided;

def lerp(a, b, t):
    if(t > 1):
        t = 1;
    if(t < 0):
        t = 0;
    return a + t*(b - a);

class GameObject: #the ultimate class for all objects to derive from
    all_gameObjects = [];
    def __init__(self, x:float, y:float, width, height, mass = 1,id = None):
        GameObject.all_gameObjects.append(self);
        self.added_momentum = np.array([0.0, 0.0]); #this is the momentum that is added to the object every frame. we update this for each force and then finally apply the velocity update.
        self.pos = np.array([x, y], dtype='float64'); #this is the position of the object in the world.
        self.pixel_x = round(self.pos[0]); 
        self.pixel_y = screen_height - round(self.pos[1]);
        self.width = width
        self.height = height
        self.halfheight = height/2;
        self.halfwidth = width/2;
        self.velocity = np.array([0.0, 0.0], dtype='float64'); #this is a vector that represents the velocity of the square.
        self.isCamera = False; #this is a boolean that determines whether or not the object is the camera.
        self.collision_layer_mask = 255;
        self.collision_layer = 1; #default collision layer. Each layer is an 8 bit value with 1 bit set to 1.
        self.elasticity = 1; # 1 means perfectly elastic, 0 means perfectly inelastic. 
        self.isStatic = False; #if this is true then the object will not move at all.
        self.mass = mass; #the type of block we have, 1 means horizontal bouncing block, and 2 means vertical bouncing block.
        #The velocity is measured in Pixels/Second.
        self.hidden = False; #if this is true then the object will not be drawn on the screen.
        if(id == None):
            self.id = len(GameObject.all_gameObjects);
        else:
            self.id = id;
    def update(self, dt):
        if(not self.hidden): 
            self.draw(screen); 

    def get_pixel_pos(self):
        return (round(self.pos[0]), round(screen_height - self.pos[1]));

    def fixed_update(self, dt): #this is where we will do all of our physics calculations.
        if(self.isStatic):
            return; #no calculation for static objects.
        #first we add the gravity force.
        #We can only do this step IF our object isn't being blocked by something below it.
        self.velocity[1] += gravity_acceleration*dt; #we add the gravity force to the velocity.
        #now we add the velocity to the position.
        self.pos += self.velocity*dt; #we add the velocity to the position.
        #self.set_position(self.pos); #we set the position to the new position.
        boundary_collision_check(self);
        pass;
    def draw(self, screen = screen):
        pass;


    def set_velocity(self, v:np.ndarray):
        self.velocity = v;
    # def set_velocity(self, x:float, y:float):
    #     self.velocity[0] = x;
    #     self.velocity[1] = y;
    def set_velocity(self, v:list):
        self.velocity[0] = v[0];
        self.velocity[1] = v[1];
    def set_position(self, x, y):
        self.pos = np.array([x, y]);
    def set_position(self, pos):
        self.pos = pos;

class particle(GameObject):
    def __init__(self, x:float, y:float, radius, mass = 1, id = None, color = (0, 40, 255)):
        super().__init__(x, y, radius, radius, mass, id);
        self.radius = radius;
        self.color = color;

    def draw(self, screen = screen):
        pygame.draw.circle(screen, self.color, self.get_pixel_pos(), self.radius);

class Camera(GameObject):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height);
        self.isCamera = True;
        self.collision_layer = 0; #the camera is on the 0th layer, so it doesn't collide with anything at all.
        self.collision_layer_mask = 0; #the camera is on the 0th layer, so it doesn't collide with anything at all.
        self.hidden = True;
        self.isStatic = False;
        self.screen_focus_pos = np.array([x, y]);
        self.set_focus_area(self.screen_focus_pos);
        self.smoothing = 5; #the smoothing factor for the camera's movement.
    def get_screen_rect(self):
        return pygame.Rect(self.x - self.width/2, self.y - self.height/2, self.width, self.height);

    def set_focus_area(self, target):
        self.x = target[0] - self.width/2;
        self.y = target[1] - self.height/2;
        self.screen_focus_pos = target;
#    def pad_reposition(self, rect:pygame.rect):
        #if the rect is going out of bounds then we shall move the camera as well, with some lerp attached to it.
    
def check_all_collisions(dt = 1/framerate):
    def handle_collision(g0, g1):
        dist = np.linalg.norm(g0.pos - g1.pos) 
        global elasticity;
        if(g0.radius + g1.radius > dist): #then we have a collision.
            #we need to calculate the normal vector of the collision.
            normal = ((g0.pos - g1.pos)/dist) ;
            g0.set_position(g0.pos + normal* (g0.radius + g1.radius - dist)/2);
            g1.set_position(g1.pos - normal* (g0.radius + g1.radius - dist)/2);

            #now we calculate the momentum changes generated, using the elasticity value.
            vel0_along_normal = np.dot(g0.velocity, normal);
            vel1_along_normal = np.dot(g1.velocity, normal);
            #now we calculate the momentum changes generated, using the elasticity value.
            v1 = vel1_along_normal; m1 = g1.mass;
            v0 = vel0_along_normal; m0 = g0.mass;
            e = elasticity;
            v1f = (v1*(m1 - e*m0) + (1+e)*m0*v0)/(m1 + m0);
            v0f = (v0*(m0 - e*m1) + (1+e)*m1*v1)/(m1 + m0);
            g0.velocity += (v0f - v0)*normal ;
            g1.velocity += (v1f - v1)*normal;
            #g0.velocity[1] -= gravity_acceleration*dt; #we simply remove the effect of gravity on the object for this frame.
            #g1.velocity[1] -= gravity_acceleration*dt;
            # print("velocities updated as", g0.velocity, g1.velocity);
            # at the same time we also remove the effect of gravity on them for this particular frame.
            return True;
        else:
            return False;
    for i in range(len(GameObject.all_gameObjects)):
        for j in range(i+1, len(GameObject.all_gameObjects)):
            if(not (GameObject.all_gameObjects[i].collision_layer & GameObject.all_gameObjects[j].collision_layer_mask)):
                continue; #if not on same collision layer we continue;
            if(GameObject.all_gameObjects[i].isStatic and GameObject.all_gameObjects[j].isStatic):
                continue;
            if(GameObject.all_gameObjects[i].isCamera or GameObject.all_gameObjects[j].isCamera):
                continue;
            g0 = GameObject.all_gameObjects[i]; g1 = GameObject.all_gameObjects[j];
            handle_collision(g0, g1);

pixel_densities = np.zeros((screen_width, screen_height));

# @jit(nopython=True)
def light_up_nearby_pixels(pixpos, radius, pixel_densities):
    #light up the nearby pixels on our screen.
    area = radius*(2.1416);
    for i in range(max(0, pixpos[0] - radius), min(screen_width, pixpos[0] + radius)):
        for j in range(max(0, pixpos[1] - radius), min(screen_height, pixpos[1] + radius)):
            dist_square = (pixpos[0] - i)**2 + (pixpos[1] - j)**2;
            dist = np.sqrt(dist_square);
            if(dist < radius):
                brightness = (50/((dist/radius)**2 + 1) - 0.5)/area; 
                pixel_densities[i][j] += brightness;

# @jit(nopython=True)
def light_all_pixels():
    global pixel_densities;
    pixel_densities.fill(0);
    for obj in GameObject.all_gameObjects:
        if(obj.hidden):
            continue;
        pixpos = obj.get_pixel_pos();
        light_up_nearby_pixels(pixpos, 1500, pixel_densities); #, lambda x: (1 - x/30));
    #print(pixel_densities.max());

main_camera = Camera(0, 0, screen_width, screen_height); #the one camera that the game will use to render objects on the screen.
GameObject.all_gameObjects.pop(); #we remove the camera from the list of gameobjects, since we do not want to loop over it.
main_camera.hidden = True; main_camera.isStatic = True; #we don't want to do physics on the camera.

playerpos = [];
print("loading world")


def get_random_particle():
    r = random.uniform(15, 35);
    p = particle(random.uniform(0, screen_width), random.uniform(0, screen_height),r, r**2);
    p.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255));
    p.set_velocity(get_random_vector2(400));
    return p;


def get_random_vector2(lim):
    return np.array([random.uniform(-lim, lim), random.uniform(-lim, lim)]);

# p1 = particle(100, 100, 50);
# p1.color = (255, 0, 0);
# p2 = particle(400, 100, 50);
# p2.color = (0, 255, 0);
# p2.set_velocity([-100, 0]);
for i in range(total_particles):
    get_random_particle();

def render_blobs(surface):
    pixarray = np.clip(pixel_densities, a_min=0, a_max=1);
    pixarray = (pixarray * 255).astype(np.uint8);
    # print(pixarray.max());
    pixarray = np.repeat(pixarray[:, :, np.newaxis], 3, axis=2)
    pygame.surfarray.blit_array(surface, pixarray)

def main():
    prev_time = pygame.time.get_ticks();
    cur_mode = 0;
    #surface = pygame.Surface((screen_width, screen_height));
    modes = ["play", "pause"];
    step_frame = 0;
    while True:
        clock.tick(physics_multiplier*framerate) #sets framerate to 60 fps
        curtime = pygame.time.get_ticks();
        dt = curtime - prev_time;
        prev_time = pygame.time.get_ticks();
        # light_all_pixels();
        # render_blobs(surface);
        if(dt != 0):
            print("FPS:", 1000/dt, "dt:", dt, "ms")
        #main_camera.set_focus_area(lerp(main_camera.screen_focus_pos, player_square.pos, 9*dt/1000));
        if(cur_mode == 0):
            for obj in GameObject.all_gameObjects:
                obj.fixed_update(1/(physics_multiplier*framerate));
            check_all_collisions();
        
        # screen.blit(surface, (0,0))
        if(step_frame%physics_multiplier == 0):
            step_frame = 0;
            screen.fill(0) #fills screen with black
            for obj in GameObject.all_gameObjects:
                obj.update(dt); #we run the update function for all our gameobjects. ALTHOUGH we should probably do this only for the main player.
        
        pygame.display.flip(); 
        step_frame += 1;
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                print("exiting pygame");
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    print("exiting pygame");
                    sys.exit()
                if event.key == K_p:
                    cur_mode = (cur_mode + 1)%len(modes); 
                    print("switched to ", modes[cur_mode]);
                    pass;
                if event.key == K_r:
                    print("resetting");
                    for obj in GameObject.all_gameObjects:
                        obj.pos[0] = random.uniform(0, screen_width);
                        obj.pos[1] = random.uniform(0, screen_height);
                        obj.velocity = get_random_vector2(400);
                    #player_square.velocity[1] += 200;
                    #player_square.velocity[0] += 200;
if __name__ == "__main__":
    start = time.time();
    main();
    pygame.quit();
    sys.exit();