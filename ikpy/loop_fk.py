import tqdm
import sys
sys.path.append("./src/")
import ikpy
import numpy as np
from ikpy import plot_utils
import matplotlib.pyplot as plt
from IPython import embed
import time

def my_sample_and_plot_work(real_frame_all, six_axis):
    to_base_end_point = np.dot(real_frame_all[-1], np.array([[0, 0, 1, 1]]).T) 
    to_base_start_point = np.dot(real_frame_all[-1], np.array([[0, 0, 0, 1]]).T) 
    to_base_vector = to_base_end_point - to_base_start_point
    to_x = np.round((180/np.pi)*np.arctan2(to_base_vector[1,0], to_base_vector[0,0]),2)
    to_z = np.round((180/np.pi)*np.arctan2(np.sqrt(to_base_vector[0,0]**2+to_base_vector[1,0]**2), to_base_vector[2,0]),2)
    pair = np.array([to_x, to_z])
    #print("***to x (longitude), to z (latitude):", to_x, to_z)
    #closest point:
    where_closest = np.abs((WANTED_PAIR_deg-pair)).sum(axis=1).argmin()
    pair_x = np.sin(WANTED_PAIR[:,1])*np.cos(WANTED_PAIR[:,0])
    pair_y = np.sin(WANTED_PAIR[:,1])*np.sin(WANTED_PAIR[:,0])
    pair_z = np.cos(WANTED_PAIR[:,1])

    #plot:
    if PLOT:
        #quiver mesh:
        ax.quiver(0,0,0, pair_x, pair_y, pair_z, linestyles='--', color='r', linewidth=0.5, alpha=0.5)
        my_chain.plot(six_axis, ax, show=True, target=[to_base_start_point, to_base_end_point])
        ax.scatter(pair_x[where_closest], pair_y[where_closest], pair_z[where_closest], color='g', marker="*")
        ax.set_xlim3d(-1.21, 1.21)
        ax.set_ylim3d(-1.21, 1.21)
        ax.set_zlim3d(-1.21, 1.21)
        ax.view_init(elev=45, azim=45)
        plt.title("Angle: %s"%np.round(six_axis,2))
        plt.draw()
        plt.pause(0.0001)
        #embed()
        #sys.exit()
        plt.cla()
    return pair, where_closest

def self_check():
    step = 30 
    angles_loop = range(0, 360+step, step)
    six_axis = np.zeros(NUM_OF_JOINTS)   #first and last won't 
    for moving_axis in range(0, len(six_axis)):
        for angle in angles_loop:
            six_axis[moving_axis] = angle*np.pi/180
            real_frame_all = np.array(my_chain.forward_kinematics(six_axis, full_kinematics=True))
            my_sample_and_plot_work(real_frame_all, six_axis)

def random_pose_sample(six_axis, pairs, densities):
    for idx,_axis in tqdm.tqdm(enumerate(six_axis), total=NUM_OF_RUNS):
        real_frame_all = np.array(my_chain.forward_kinematics(_axis, full_kinematics=True))
        pair, density = my_sample_and_plot_work(real_frame_all, _axis)
        pairs[idx] = pair
        densities[idx] = density
    return pairs, densities

if __name__ == "__main__":
    print("start...")

    #Get physical chain:
    my_chain = ikpy.chain.Chain.from_urdf_file("./resources/BA006N.urdf")

    #CONFIGS:
    NUM_OF_JOINTS = 7
    NUM_OF_RUNS = int(1e7)
    step = 30
    to_z_mesh = range(0, 180+step, step)
    to_x_mesh = range(-180, 180, step)
    TO_Z_MESH, TO_X_MESH = np.meshgrid(to_z_mesh, to_x_mesh)
    WANTED_PAIR_deg = np.array([TO_X_MESH.reshape(-1), TO_Z_MESH.reshape(-1)]).T
    WANTED_PAIR = np.array([TO_X_MESH.reshape(-1), TO_Z_MESH.reshape(-1)]).T*np.pi/180
    PLOT = True
    PLOT = False
    plt.ion()
    ax = plot_utils.init_3d_figure()

    #Check first:
    self_check()
    #prepare:
    real_frames = np.empty(shape=(NUM_OF_RUNS, NUM_OF_JOINTS,4,4))
    six_axis = np.random.random((NUM_OF_RUNS, NUM_OF_JOINTS))*720-360
    pairs = np.empty(shape=(NUM_OF_RUNS, 2))
    densities = np.empty(shape=(NUM_OF_RUNS, 1))
    #and run:
    pairs, densities = random_pose_sample(six_axis, pairs, densities)

    #For output pose
    nearest_to_anchor = []
    for i in WANTED_PAIR_deg: 
        nearest_to_anchor.append(np.abs((pairs-i)).sum(axis=1).argmin())
    WANTED_POSE = six_axis[nearest_to_anchor]
    np.savetxt("joint_positions.txt", WANTED_POSE)
    np.savetxt("density.txt", densities)



