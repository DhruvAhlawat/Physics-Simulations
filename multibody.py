import pygame
import sys
import os
import numpy as np;
from pygame.locals import *
import time
import pickle
import random
import argparse
from PIL import Image
# import numba 
# from numba import jit
pygame.mixer.pre_init(44100, -16, 2, 512) # setup mixer to avoid sound lag
pygame.init() #initializes pygame

## USAGE: python3 multibody.py -n [total_particles] -r [mean_radius]
## ESC to quit, P to pause, R to reset, G to toggle gravity, I to toggle particle creation, C to toggle dynamic particle colors (speed based). 
## Right arrow key to step one (display)frame ahead when paused.
physics_multiplier = 2; ## NOTE: If lagging occurs, reduce this parameter to 1. or otherwise reduce framerate below its current value. 
## NOTE: this is basically the number of physics steps per frame. on increasing this value, we get more accurate physics.

show_velocities = True; #if this is true then the velocities of the particles will be shown as a line from the center.
enable_dynamic_colors = True; #if this is true then the colors of the particles will change based on their velocity.
enable_gravity = False;
framerate = 64; #sets framerate to 64 FPS. A power of 2 ensures 1/framerate is exact.
normal_gravitational_acceleration = -1150;
elasticity =  0.9 #means perfectly elastic, 0 means perfectly inelastic.
if(enable_gravity and elasticity == 1):
    elasticity = 0.9;
## SUGGESTION: If gravity is on, elasticity should be ideally not 1, Otherwise I recommend using elasticity as 1 or slightly greater to not let the energy be lost from the particles
slow_color = np.array([30,100,255]); fast_color = np.array([255, 40, 0]);
particle_radius_range = (5,25); 
mean_radius = 15; std_radius = 10;
clock = pygame.time.Clock()
screen_width = 720; 
screen_height = 720;
screen = pygame.display.set_mode((screen_width, screen_height)) #creates screen
pygame.mouse.set_visible(0);
# bg = pygame.image.load("./images/bubble2_large.jpg") #loads background image
fixed_delta_seconds = 1/framerate;
total_particles = 100;
#FORCE VARIABLES TO CONSIDER:
if(enable_gravity == False):
    gravity_acceleration = 0;
else:
    gravity_acceleration = normal_gravitational_acceleration;
#BOUNDING BOX DETAILS:
#the bounding box is a rectangle that is aligned with the x and y axes.
lower_boundary = 0;
upper_boundary = screen_height;
left_boundary = 0;
right_boundary = screen_width;


## Additional Utils:
mousepos = pygame.mouse.get_pos();
ForceType = 0; #0 means no force, 1 means inwards force, 2 means outwards force
#Left click is for inwards force, and right click is for outwards force.

parser = argparse.ArgumentParser(description='arguments for generation of particles')
parser.add_argument('-n', '--total_particles', type=int, default=30, help='total number of particles to be generated')
parser.add_argument('-r', '--mean_radius', type=int, default=15, help='mean radius of the particles')
args = parser.parse_args()
mean_radius = args.mean_radius;
total_particles = args.total_particles;
particle_radius_range = (mean_radius - 5, mean_radius);
def boundary_collision_check(obj):
    collided = False;
    if(obj.pos[1] - obj.radius < lower_boundary):
        obj.pos[1] = lower_boundary + obj.radius;
        obj.velocity[1] = - elasticity * obj.velocity[1];
        collided = True;

    if(obj.pos[1] + obj.radius > upper_boundary):
        obj.pos[1] = upper_boundary - obj.radius;
        obj.velocity[1] = - elasticity * obj.velocity[1];
        # collided = True;

    if(obj.pos[0] - obj.radius < left_boundary):
        obj.pos[0] = left_boundary + obj.radius;
        obj.velocity[0] = - elasticity * obj.velocity[0];
        # collided = True;

    if(obj.pos[0] + obj.radius > right_boundary):
        obj.pos[0] = right_boundary - obj.radius;
        obj.velocity[0] = - elasticity * obj.velocity[0];
        # collided = True;
    return collided;

def lerp(a, b, t):
    if(t > 1):
        t = 1;
    if(t < 0):
        t = 0;
    return a + t*(b - a);

grid_size = particle_radius_range[1] * 2; #the size of the grid is exactly equal to the maximum diameter of the particles
total_grids = []; 
for i in range(int(screen_width/grid_size) + 1):
    total_grids.append([]);
    for j in range(int(screen_height/grid_size) + 1):
        total_grids[i].append(set());

# returns the index of the grid that the point (x,y) is in.
def grid_index_hash(x, y):
    return (int(x/grid_size), int(y/grid_size)); #currently this is just the grid number that the particle is in.

def move_particle_cell(id, prev_index, new_index):
    total_grids[prev_index[0]][prev_index[1]].remove(id);
    total_grids[new_index[0]][new_index[1]].add(id);

def nearby_indices(index):
    x,y = index;
    for i in range(-1,2):
        for j in range(-1,2):
            if(x+i < 0 or y+j < 0 or x+i >= len(total_grids) or y+j >= len(total_grids[0])):
                continue;
            yield (x+i, y+j);

class GameObject: #the ultimate class for all objects to derive from
    all_gameObjects = [];
    def __init__(self, x:float, y:float, width, height, mass = 1,id = None):
        GameObject.all_gameObjects.append(self);
        self.id = len(GameObject.all_gameObjects) - 1; #we give its id as its index in the list.
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
        self.cell_index = grid_index_hash(x,y); #this is the cell index that the object is in.
        self.collided_this_frame = False;
        #The velocity is measured in Pixels/Second.
        self.hidden = False; #if this is true then the object will not be drawn on the screen.
    def update(self, dt):
        if(not self.hidden): 
            self.draw(screen); 

    def get_pixel_pos(self):
        return (round(self.pos[0]), round(screen_height - self.pos[1]));
    
    def get_screenpos(pos):
        return (round(pos[0]), round(screen_height - pos[1]));

    def fixed_update(self, dt): #this is where we will do all of our physics calculations.
        if(self.isStatic):
            return; #no calculation for static objects.
        #first we add the gravity force.
        #We can only do this step IF our object isn't being blocked by something below it.
        #need to add force from the mouse here.
        if(ForceType == 0 and self.collided_this_frame == False and enable_gravity == True):
                self.velocity[1] += gravity_acceleration*dt; #we add the gravity force to the velocity.
        elif(ForceType == 1):
            #then we need to add a force towards the mouse.
            mousepos = pygame.mouse.get_pos();
            mousepos = np.array([mousepos[0], screen_height - mousepos[1]]);
            dist = np.linalg.norm(mousepos - self.pos);
            if(dist == 0):
                return;
            direction = (mousepos - self.pos)/dist;
            if(dist < 100):
                self.velocity += direction*1000*dt/np.sqrt(dist);
            elif(not self.collided_this_frame):
                self.velocity[1] += gravity_acceleration*dt; #we add the gravity force to the velocity.
        #now we add the velocity to the position.
        self.pos += self.velocity*dt; #we add the velocity to the position.
        #self.set_position(self.pos); #we set the position to the new position.
        boundary_collision_check(self);
        new_cell_index = grid_index_hash(self.pos[0], self.pos[1]);
        if(new_cell_index != self.cell_index):
            #then we need to update the cell index of the object.
            move_particle_cell(self.id, self.cell_index, new_cell_index);
            self.cell_index = new_cell_index;
        #Now based on these details we decide which cell position our object actually is in.

        #print(self.id, self.pos);
        if(np.isnan(self.pos[0]) or np.isnan(self.pos[1])):
            print("ERROR: NAN POSITION");
            print(self.id, self.pos);
            raise Exception("NAN POSITION");
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
        self.pos = np.array([x, y], dtype='float64');
    def set_position(self, pos):
        self.pos = np.array(pos, dtype='float64');

class particle(GameObject):
    def __init__(self, x:float, y:float, radius, mass = 1, color = (0, 40, 255)):
        super().__init__(x, y, radius, radius, mass);
        self.radius = radius;
        self.color = color;
        total_grids[self.cell_index[0]][self.cell_index[1]].add(self.id); #we add the id of the particle to the grid that it is in.

    def draw(self, screen = screen):
        if(enable_dynamic_colors):
            velmag = np.linalg.norm(self.velocity); velmag = max(0, velmag);
            self.color = tuple(lerp(slow_color, fast_color, np.sqrt(velmag)/35));
        pixpos = self.get_pixel_pos();
        pygame.draw.circle(screen, self.color, pixpos, self.radius);
        #Then we will draw an arrow corresponding to its velocity
        if(show_velocities and cur_mode == 1):
            vel = self.velocity.copy(); vel[1] = -vel[1];
            vel = np.round(1.5*self.radius*vel/np.linalg.norm(vel));
            pygame.draw.line(screen, (255 - self.color[0], 255 - self.color[1], 255 - self.color[2]), pixpos, pixpos + vel, 2);

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
        if(dist == 0):
            return False;
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
    cnt = 0;
    for i in range(len(GameObject.all_gameObjects)):
        GameObject.all_gameObjects[i].collided_this_frame = boundary_collision_check(GameObject.all_gameObjects[i]); #setting their collision values to this.
    for i in range(len(GameObject.all_gameObjects)):
        cellindex = GameObject.all_gameObjects[i].cell_index; #we get its cell index first. then we check all the objects in the same cell and adjacent cells.
        for (x,y) in nearby_indices(cellindex):
            for j in total_grids[x][y]:
                if(i == j):
                    continue; #same object.
                cnt += 1;
                if(i > len(GameObject.all_gameObjects) or j > len(GameObject.all_gameObjects)):
                    print("ERROR\n\n");
                    raise Exception("Index out of bounds");
                # print("checking collision between", i, "and", j, " total: ", len(GameObject.all_gameObjects));
                if(not (GameObject.all_gameObjects[i].collision_layer & GameObject.all_gameObjects[j].collision_layer_mask)):
                    continue;
                if(GameObject.all_gameObjects[i].isStatic and GameObject.all_gameObjects[j].isStatic):
                    continue;
                if(GameObject.all_gameObjects[i].isCamera or GameObject.all_gameObjects[j].isCamera):
                    continue;
                g0 = GameObject.all_gameObjects[i]; g1 = GameObject.all_gameObjects[j];
                collided = handle_collision(g0, g1);
                # if(collided):
                #     g0.collided_this_frame = True;
                #     g1.collided_this_frame = True; #we set their collisions to true.
        # for j in range(i+1, len(GameObject.all_gameObjects)):
        #     if(not (GameObject.all_gameObjects[i].collision_layer & GameObject.all_gameObjects[j].collision_layer_mask)):
        #         continue; #if not on same collision layer we continue;
        #     if(GameObject.all_gameObjects[i].isStatic and GameObject.all_gameObjects[j].isStatic):
        #         continue;
        #     if(GameObject.all_gameObjects[i].isCamera or GameObject.all_gameObjects[j].isCamera):
        #         continue;
        #     g0 = GameObject.all_gameObjects[i]; g1 = GameObject.all_gameObjects[j];
        #     handle_collision(g0, g1);
    # print("total collisions checked:", cnt);
# pixel_densities = np.zeros((screen_width, screen_height));

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
main_camera.hidden = True; main_camera.isStatic = True; #we don't want to do physics on the camera.
GameObject.all_gameObjects.remove(main_camera); #we remove the camera from the list of gameobjects. 

print("loading world")
def get_random_particle():
    r = random.gauss(mean_radius, std_radius);
    r = max(r, particle_radius_range[0]); r = min(r, particle_radius_range[1]);
    p = particle(random.uniform(0, screen_width), random.uniform(0, screen_height),r, r**2);
    p.color = get_random_color();
    p.set_velocity(get_random_vector2(400));
    return p;

def get_random_vector2(lim):
    return np.array([random.uniform(-lim, lim), random.uniform(-lim, lim)]);

def get_random_color():
    c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255));
    normc = c[0]**2 + c[1]**2 + c[2]**2;
    normc = np.sqrt(normc);
    c = (255 * c[0]//normc, 255 * c[1]//normc, 255 * c[2]//normc);
    return c;


def shoot_particle(init_pos, init_vel, radius, mass, color):
    p = get_random_particle(); p.set_position(init_pos);
    p.set_velocity(init_vel);


# ## Create randomized particles here.
# for i in range(total_particles):
#     get_random_particle();

print("total gameobjects:", len(GameObject.all_gameObjects));
def render_blobs(surface):
    pixarray = np.clip(pixel_densities, a_min=0, a_max=1);
    pixarray = (pixarray * 255).astype(np.uint8);
    # print(pixarray.max());
    pixarray = np.repeat(pixarray[:, :, np.newaxis], 3, axis=2)
    pygame.surfarray.blit_array(surface, pixarray)

class PygameRecord:
    def __init__(self, filename: str, fps: int):
        self.fps = fps
        self.filename = filename
        self.frames = []

    def add_frame(self):
        curr_surface = pygame.display.get_surface()
        x3 = pygame.surfarray.array3d(curr_surface)
        x3 = np.moveaxis(x3, 0, 1)
        array = Image.fromarray(np.uint8(x3))
        self.frames.append(array)

    def save(self):
        self.frames[0].save(
            self.filename,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / self.fps),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred: {exc_value}")
        self.save()
        # Return False if you want exceptions to propagate, True to suppress them
        return False

cur_mode = 0;
pygame.mouse.set_visible(1);
def main():
    global enable_gravity, gravity_acceleration, normal_gravitational_acceleration, elasticity, enable_dynamic_colors;
    global ForceType;
    global mousepos;
    global total_particles;
    global cur_mode;
    creating_particles = False;
    prev_time = pygame.time.get_ticks();
    #surface = pygame.Surface((screen_width, screen_height));
    modes = ["play", "pause"];
    step_frame = 0; played_next_step = False;
    delay_between_shooting = 4; cur_shoot_frame = 0;
    particles_left = total_particles;
    recorder = PygameRecord(f"gifs/output_{time.time()}.gif", 60);
    while True:
        clock.tick(physics_multiplier*framerate) #sets framerate to 60 fps
        if(cur_mode == 0):
            if(cur_shoot_frame == delay_between_shooting and step_frame%physics_multiplier == 0):
                cur_shoot_frame = 1;
                if(particles_left > 0 or creating_particles == True): #we shoot out balls from the top if creating particles is true.
                    shoot_particle((mean_radius*3, screen_height), np.array([600,-200]), mean_radius, 1, (255, 0, 0));
                if(particles_left > 0):
                    particles_left -= 1;
            elif(step_frame%physics_multiplier == 0):
                cur_shoot_frame += 1;
        curtime = pygame.time.get_ticks();
        dt = curtime - prev_time;
        prev_time = pygame.time.get_ticks();
        # light_all_pixels();
        # render_blobs(surface);
        if(dt != 0 and step_frame%physics_multiplier == 0):
            print("FPS:", 1000/(dt*physics_multiplier), "dt:", dt*physics_multiplier, "ms")
        #main_camera.set_focus_area(lerp(main_camera.screen_focus_pos, player_square.pos, 9*dt/1000));
        mousepos = pygame.mouse.get_pos();
        if(cur_mode == 0):
            # check_all_collisions(1/(physics_multiplier*framerate));
            for obj in GameObject.all_gameObjects:
                # boundary_collision_check(obj);
                obj.fixed_update(1/(physics_multiplier*framerate));
            check_all_collisions(1/(physics_multiplier*framerate));
        # print("max velocity:", max([np.linalg.norm(obj.velocity) for obj in GameObject.all_gameObjects]));
        # screen.blit(surface, (0,0))
        if(played_next_step == True and step_frame == physics_multiplier):
            played_next_step = False;
            cur_mode = 1; #pausing the game again after rendering the next frame. 
        if(step_frame%physics_multiplier == 0):
            step_frame = 0;
            screen.fill(0) #fills screen with black
            for obj in GameObject.all_gameObjects:
                obj.update(dt); #we run the update function for all our gameobjects. ALTHOUGH we should probably do this only for the main player.
        
        pygame.display.flip(); 
        step_frame += 1;
        recorder.add_frame(); #we add the frame to the recorder.
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
                if event.key == K_s:
                    print("saving recording");
                    recorder.save();
                if event.key == K_q:
                    print("Starting recording");
                    recorder = PygameRecord(f"gifs/output_{time.time()}.gif", 60);
                if event.key == K_RIGHT:
                    if(cur_mode == 1): #Then we need to step one frame ahead. 
                        played_next_step = True;
                        cur_mode = 0;
                        pass;
                if event.key == K_g:
                    enable_gravity = not enable_gravity;
                    print("gravity is now", enable_gravity);
                    if(enable_gravity == False):
                        gravity_acceleration = 0;
                        elasticity = 0.995;
                    else:
                        elasticity = 0.9;
                        gravity_acceleration = normal_gravitational_acceleration;
                if event.key == K_i:
                    creating_particles = not creating_particles;
                    print("creating particles is now", creating_particles);
                    print("total particles: ", len(GameObject.all_gameObjects) - 1);
                if event.key == K_c:
                    enable_dynamic_colors = not enable_dynamic_colors;
                    print("dynamic colors is now", enable_dynamic_colors);
                    if(enable_dynamic_colors == False):
                        for obj in GameObject.all_gameObjects:
                            obj.color = get_random_color();
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    ForceType = 1;
                    print("left click")
                if event.button == 3:
                    ForceType = 2;
                    print("right click")
                mousepos = pygame.mouse.get_pos();
            if event.type == MOUSEBUTTONUP:
                mousepos = pygame.mouse.get_pos();
                ForceType = 0; #no force towards the mouse.
                pass;
if __name__ == "__main__":
    start = time.time();
    main();
    pygame.quit();
    sys.exit();