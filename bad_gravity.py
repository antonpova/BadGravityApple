'''
hello! this code renders "Bad Apple!!" as a bunch of particles.
it definitely isn't perfect and relies on some magic numbers, 
but feel free to use it as inspiration and experiment with it!

Please mention the author (anton pova) if you found this useful <3
'''



import cv2
import numpy as np
from numba import njit, prange


#config
#render settings
scale = 2
w=480*scale
h=360*scale
fps=30
count = 300 #the number of frames of the simulation, use 6572 for the whole video (be careful though, the full file have a size 784MB for me)
fade=.5 #used for trail effect

#simulation settings
speed = 1
speed_repel = 50
slowdown=2.0
max_attractors=10000000 
num = 480*2+360*2-4

vid_source = cv2.VideoCapture("bad_apple.mp4")

@njit(parallel=True, fastmath=True)
def update_physics_numba(pos, vel, attractors, speed, speed_repel, slowdown):
    num_particles = len(pos)
    num_attractors = len(attractors)
    
    for i in prange(num_particles):
        px = pos[i, 0]
        py = pos[i, 1]
        
        force_x = 0.0
        force_y = 0.0
        
        if num_attractors > 0:
            for j in range(num_attractors):
                dx = attractors[j, 0] - px
                dy = attractors[j, 1] - py
                
                r2 = dx*dx + dy*dy + 3.0
                factor = speed / r2
                
                force_x += dx * factor
                force_y += dy * factor
        
        new_vx = (vel[i, 0] / slowdown) + force_x
        new_vy = (vel[i, 1] / slowdown) + force_y

        repel_x = 0.0
        repel_y = 0.0
        
        for j in range(num_particles):
            dx = pos[j, 0] - px
            dy = pos[j, 1] - py
            
            r2 = dx*dx + dy*dy + 1.0 
            r = np.sqrt(r2)
            factor = speed_repel / (r2 * r)
            
            repel_x += dx * factor
            repel_y += dy * factor
            
        new_vx -= repel_x
        new_vy -= repel_y
        
        vel[i, 0] = new_vx
        vel[i, 1] = new_vy

    for i in prange(num_particles):
        pos[i, 0] += vel[i, 0]
        pos[i, 1] += vel[i, 1]
        
    return pos, vel


def filter_by_density(mask, window_size=15, max_points_per_window=5):
    #square mask
    #density = cv2.boxFilter(mask, cv2.CV_32F, (window_size, window_size), normalize=False)

    #circle mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window_size, window_size))
    density = cv2.filter2D(mask, cv2.CV_32F, kernel)
    
    density[density == 0] = 1.0
    
    target_sum = max_points_per_window * 255.0
    probability_map = target_sum / density
    random_roll = np.random.uniform(0.0, 1.0, mask.shape)
    
    kept_pixels = (mask > 0) & (random_roll < probability_map)
    y_idx, x_idx = np.where(kept_pixels)
    if len(x_idx) > 0:
        attractors = np.column_stack((x_idx, y_idx)).astype(np.float32)
    else:
        attractors = np.zeros((0, 2), dtype=np.float32)
            
    return attractors


def main():
    #x y vx vy
    particle_pos = np.zeros((num, 2), np.float32)
    particle_vel = np.zeros((num, 2), np.float32)
    #particle_weight = np.ones((num), np.float32)

    r=min(w,h)*.4

    v=1
    for i in range(num):
        if i<480:
            particle_pos[i]=[i*scale,0]
        elif i<(480+360):
            particle_pos[i]=[w-1,(i-480)*scale]
        elif i<(480+360+480):
            particle_pos[i]=[(i-480-360)*scale,h-1]
        else:
            particle_pos[i]=[0,(i-480-360-480)*scale]
        
        
        

    img = np.zeros((h,w,3), np.float32)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("test.mp4", fourcc, fps, (w, h), isColor=True)


    for frame in range(count):
        img*=fade
        ret, frame_source = vid_source.read()
        gray = cv2.cvtColor(frame_source, cv2.COLOR_BGR2GRAY)

        #original vid
        #_, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        #only edges
        mask = cv2.Canny(gray, 100, 200)

        
        y_idx, x_idx = np.where(mask > 0)
        
        attractors = filter_by_density(mask, window_size=25, max_points_per_window=20)
        attractors*=scale

        if len(attractors) > max_attractors:
            indices = np.random.choice(len(attractors), max_attractors, replace=False)
            attractors = attractors[indices]

        particle_pos, particle_vel = update_physics_numba(
                particle_pos, 
                particle_vel, 
                attractors, 
                float(speed), 
                float(speed_repel), 
                float(slowdown)
        )

            
        
        
        for i in range(num):
            cv2.circle(img, particle_pos[i].astype(np.int32), scale, (0,1,0), -1)
        
        #out.write((img* 255).astype(np.uint8))
        
        img_copy=img.copy()
        cv2.putText(img_copy, f"T: {frame/30:.2f}s", (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
        out.write(cv2.convertScaleAbs(img_copy, alpha=255))
            
        cv2.imshow("preview", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(mask.shape)
            break
        
    out.release()



if __name__ == '__main__':
    #if you don't need the profiler:
    main()

    #else:
    #import cProfile
    #import pstats
    #cProfile.run('main()', sort='cumtime')
