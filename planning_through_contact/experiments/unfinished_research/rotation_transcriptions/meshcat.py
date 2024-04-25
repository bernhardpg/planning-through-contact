# import meshcat
# import meshcat.transformations as tf
# from meshcat.animation import Animation
# from meshcat.geometry import Box
#
# vis = meshcat.Visualizer()
#
# vis["box1"].set_object(Box([0.1, 0.2, 0.3]))
#
# anim = Animation(default_framerate=1)
#
# with anim.at_frame(vis, 0) as frame:
#     # `frame` behaves like a Visualizer, in that we can
#     # call `set_transform` and `set_property` on it, but
#     # it just stores information inside the animation
#     # rather than changing the current visualization
#     frame["box1"].set_transform(tf.translation_matrix([0, 0, 0]))
# with anim.at_frame(vis, 1) as frame:
#     frame["box1"].set_transform(tf.translation_matrix([0, 1, 0]))
#
# # `set_animation` actually sends the animation to the
# # viewer. By default, the viewer will play the animation
# # right away. To avoid that, you can also pass `play=false`.
# vis.set_animation(anim)
#
# breakpoint()
#
#
#

from __future__ import absolute_import, division, print_function

import math
import time

import meshcat

vis = meshcat.Visualizer().open()

box = meshcat.geometry.Box([0.5, 0.5, 0.5])
vis.set_object(box)

draw_times = []

vis["/Background"].set_property("top_color", [1, 0, 0])

for i in range(200):
    theta = (i + 1) / 100 * 2 * math.pi
    now = time.time()
    vis.set_transform(meshcat.transformations.scale_matrix(i))
    draw_times.append(time.time() - now)
    time.sleep(1)

print(sum(draw_times) / len(draw_times))
