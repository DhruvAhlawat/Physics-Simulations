# Physics Simulations
## Introduction
I wrote this code just for fun, and have learnt many things along the way. I plan to include various different forms of physics simulations. 
1. the multibody.py file runs a simulation of a multibody collision system. There are various ways to interact with the running simulation and I've added instructions as comments (I have hope some other people check it out too just for fun).
Usage: 
    > ESC to quit, P to pause, R to reset, G to toggle gravity, C to toggle Dynamic colors (speed based), I to toggle particle creation. Right arrow key to step one (display)frame ahead when paused.

lastly I plan to add some gifs of the simulations

### Update
I added gifs for the simulations! Now this page doesn't look so dull. Also added gif creation controls
Gif usage:

    press 'Q' to start recording a gif
    press 'S' to save the gif. will pause the current simulation display while this proceeds. saves to a file with a unique name in "gifs" folder

# GIFs :

![gravity then no gravity](https://github.com/DhruvAhlawat/Physics-Simulations/blob/e5394c8e140de63e47224ddd69511ad24b271dff/gifs/gravitiy_drop.gif)

**showing the toggling of "Gravity" and "color" modes on and off **
---
![Pause and step-by-steo display](https://github.com/DhruvAhlawat/Physics-Simulations/blob/e5394c8e140de63e47224ddd69511ad24b271dff/gifs/pause_display.gif)

**Pause and step-by-step movement display. Includes direction of velocity**

---
![Mouse hold and pick](https://github.com/DhruvAhlawat/Physics-Simulations/blob/e5394c8e140de63e47224ddd69511ad24b271dff/gifs/mouse_pick.gif)

**group of balls can be picked up by holding the mouse and dragging**

# Further steps

With the realisation that threading does not work on python due to the GIL, I will move this to a C++ framework, or integrate CUDA programming into this code although I haven't worked with CUDA on python before. But let's see how it turns out. 
