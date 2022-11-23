import ParticleClass as pc
import numpy as np

import os
from matplotlib import cm
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(998)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)

def pause():  # fonction qui peut vous être utile pour debugger
    ProgramPause = input("Press the <ENTER> key to continue...")


'''Initialize system with following parameters'''
NumberOfMonomers = 50
L_xMin, L_xMax = 0, 20
L_yMin, L_yMax = 0, 20
NumberMono_per_kind = np.array([ 50])
Radiai_per_kind = np.array([ 0.2])
Densities_per_kind = np.array([ 1.0])

k_BT = 10

'''define parameters for MD simulation'''
total_time = 0.0
dt_frame = 0.02
NumberOfFrames = 100
StateOfParticle = [[1,49,0] ]#1st column for the "sick particules" 2nd colum for the "healthy particules" and 3rd column for the "recovered particules"
# call constructor, which should initialize the configuration
mols = pc.Monomers(NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT, NumberOfFrames, dt_frame )
    
mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')
#we could initialize next_event, but it's not necessary
#next_event = pc.CollisionEvent( Type = 'wall or other, to be determined', dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)



'''define parameters for MD simulation'''

next_event = mols.compute_next_event()

dummyArray = np.zeros( NumberOfFrames )
def MolecularDynamicsLoop( frame ):
    '''
    The MD loop including update of frame for animation.
    '''
    global total_time, mols, next_event
    
    #--> write your event-driven loop
    #--> we want a simulation in constant time steps dt
    next_frame_time = total_time + dt_frame
    future_time_speed_change = total_time+next_event.dt
    
    while next_frame_time >= future_time_speed_change :
        mols.pos += mols.vel * next_event.dt
        total_time += next_event.dt
        mols.compute_new_velocities(next_event)
        mols.compute_new_color(next_event, total_time) #total_time
        next_event = mols.compute_next_event()
        future_time_speed_change = total_time+next_event.dt

        for k in range(mols.NM): #hihi
            if mols.time_recover[k] < total_time :
                mols.compute_recover(k)
                mols.time_recover[k] = NumberOfFrames * dt_frame + 1

        Nb_sick = 0
        Nb_healthy = 0
        Nb_recover = 0
        for k in mols.color :
            if k == 0.2 :
                Nb_sick += 1
            elif k == 0.8 :
                Nb_healthy += 1
            else :
                Nb_recover += 1
        StateOfParticle.append([Nb_sick, Nb_healthy, Nb_recover])
    dummyArray[frame] = frame*dt_frame

    dtr = next_frame_time - total_time
    mols.pos += mols.vel * (dtr)
    total_time = next_frame_time
    next_event.dt -= dtr
    

    # we can save additional snapshots for debugging -> slows down real-time animation
    #mols.snapshot( FileName = Snapshot_output_dir + '/Conf_t%.8f_0.png' % total_time, Title = '$t = %.8f$' % total_time)
    
    plt.title( '$t = %.4f$, remaining frames = %d' % (total_time, NumberOfFrames-(frame+1)) )
    collection.set_offsets( mols.pos )
    MonomerColors = mols.color
    collection.set_array(MonomerColors)
    return collection




'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin #not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax #not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')


# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)


# plotting all monomers as solid circles of individual color


Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
MonomerColors = mols.color
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)
'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()
#A = MolecularDynamicsloop

StateParticle = np.array(StateOfParticle)
print(StateParticle[:,0])
print(StateParticle)
plt.figure
plt.plot(StateParticle[:,0], label = 'sick particles')
plt.plot(StateParticle[:,1], label ='healthy particles')
plt.plot(StateParticle[:,2], label = 'recovered particles')
plt.plot(dummyArray, label = 'time over nr of frame')
plt.xlabel('Number of Frames')
plt.ylabel('Number of Particles')
plt.title('Evolution of the state of the particles') 
plt.legend()
plt.show()

'''Save the final configuration and make a snapshot.'''
#write the function to save the final configuration
#mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % total_time)
