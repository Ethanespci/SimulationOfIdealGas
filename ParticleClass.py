import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle

            
class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        mono_1 = index of monomer
        mono_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, mono_1 = {:d}, w_dir = {:d}".format(self.Type, self.dt, self.mono_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, mono_1 = {:d}, mono_2 = {:d}".format(self.Type, self.dt, self.mono_1, self.mono_2)

class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without overlap)
    and velocities of random orientation and norms satisfying
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
    """
    def __init__(self, NumberOfMonomers = 50, L_xMin = 0, L_xMax = 20, L_yMin = 0, L_yMax = 20, NumberMono_per_kind = np.array([50]), Radiai_per_kind = np.array([ 0.2]), Densities_per_kind = np.array([ 1.0]), k_BT = 10,  NumberOfFrames = 600, dt_frame = 0.02, Lr_xMin = 4 , Lr_xMax =6 , Lr_yMin =4 , Lr_yMax =16 ):
        assert ( NumberOfMonomers > 0 )
        assert ( (L_xMin < Lr_xMin < Lr_xMax < L_xMax) and  ( L_yMin < Lr_yMin < Lr_yMax < L_yMax)) #we assert that or object is countained in the box and that the different lengths are well defined
        self.NM = NumberOfMonomers
        self.DIM = 2 #dimension of system
        self.BoxLimMin = np.array([ L_xMin, L_yMin])
        self.BoxLimMax = np.array([ L_xMax, L_yMax])
        self.mass = np.empty( self.NM ) # Masses, not initialized but desired shape
        self.rad = np.empty( self.NM ) # Radiai, not initialized but desired shape
        self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
        self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
        self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
        self.next_wall_coll = CollisionEvent( Type = 'wall', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
        self.next_mono_coll = CollisionEvent( Type = 'mono', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 0)
        self.color = np.empty(self.NM) #Color, not initialized but desired shape
        self.time_recover = np.empty(self.NM) #time after which the particle goes from sick to recovered, not initialized but desired shape
        self.objetgauche = np.array([Lr_xMin, Lr_yMin]) #In this vector there is the position of the left wall and the bottom wall of the object
        self.objetdroit = np.array([Lr_xMax, Lr_yMax]) #In this vector there is the position of the right wall and the top wall of the object
        
        self.assignRadiaiMassesVelocities( NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )   #we execute the functions we have coded after
        self.assignTimeRecover(NumberOfFrames, dt_frame)
        self.assignRandomMonoPos( )

        

    
    def assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([50]), Radiai_per_kind = np.array([ 0.2]), Densities_per_kind = np.array([ 1.0]), k_BT = 10):
        
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        '''
        
        '''initialize radiai and masses'''
        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))
        
        #-->> your turn
        #Monomers can be initialized with individual radiai and density = mass/volume.
        #For example:
        #NumberOfMonomers = 7
        #NumberMono_per_kind = np.array([ 2, 5])
        #Radiai_per_kind = np.array([ 0.2, 0.5])
        #Densities_per_kind = np.array([ 2.2, 5.5])
        #then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
        #and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
        index = 0              
        for i in range(NumberMono_per_kind.shape[0]) :                            #we define the radius and mass of our particles 
            for j in range(NumberMono_per_kind[i]) :
                self.rad[index] = Radiai_per_kind[i]
                self.mass[index] = Densities_per_kind[i] * np.pi * self.rad[j]**2
                index += 1
    
    
        
        
        '''initialize velocities'''
        
        assert( k_BT > 0 )
        # E_kin = sum_i m_i /2 v_i^2 = N * dim/2 k_BT https://en.wikipedia.org/wiki/Ideal_gas_law#Energy_associated_with_a_gas


        
        #-->> your turn
        #Initial configuration of $N$ monomers has velocities of random
        #orientation and norms satisfying
        #$E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
        #$k_B$ the Boltzmann constant, and $T$ the temperature.
        Ec_part = k_BT
        for k in range(self.NM) :
            Norm_v_part = np.sqrt(2*Ec_part/self.mass[k])           
            theta = np.random.uniform(0,2*np.pi)
            self.vel[k] = Norm_v_part * np.cos(theta), Norm_v_part * np.sin(theta)

#alternativ
#Energie_part = [np.sqrt(np.random.uniform(0,Ec))]
#       sum_energie = Energie_part[0]
#       for i in range (1,self.NM-1):
#           Energie_part.append(np.random.uniform(0, Ec - sum_energie))
#
#           sum_energie += Energie_part[i]
#       Energie_part.append(Ec - sum_energie)
#
#       norme_vitesse = []
#       for k in range(len(Energie_part)) :
#           norme_vitesse.append(np.sqrt(2*Energie_part[k]/self.mass[k]))
#
#       for i in range(self.NM):
#           self.vel[i][0] = np.random.uniform(-norme_vitesse[i], norme_vitesse[i])
#           self.vel[i][1] = np.sqrt(norme_vitesse[i]**2 - self.vel[i][0]**2)
#
        print( 'Difference between the real Energy of our particles and N*K_BT : ',((self.vel**2).sum(1) * self.mass / 2 ).sum() - self.NM * k_BT ) 


#initialize color

        for k in range(self.NM):
            self.color[k] = 0.8   #at the beginning, all particles are healthy whithout one of them
        self.color[0] = 0.2 #the first particle is chosen sick 
        
        
    def assignTimeRecover(self, NumberOfFrames = 600, dt_frame = 0.02): #we define the time when the particle is recovered after being sick
        self.time_recover[0] = 4
        for k in range(1,self.NM):
            self.time_recover[k] = NumberOfFrames * dt_frame +1 #we calculate the final total time and add 1 to it in order to be sure that during the loop we never exceed this final total time.

        
    def assignRandomMonoPos(self, start_index = 0 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert ( min(self.rad) > 0 )#otherwise not initialized
        mono_new, infiniteLoopTest = start_index, 0
        while mono_new < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            self.pos[mono_new] = np.random.uniform(self.BoxLimMin[0] + self.rad[mono_new],self.BoxLimMax[0] - self.rad[mono_new]), np.random.uniform(self.BoxLimMin[1] + self.rad[mono_new],self.BoxLimMax[1] - self.rad[mono_new])
            radius_mono_new = self.rad[mono_new]
            k, NoOverlap = 0, True
            image_droit = self.objetdroit + self.rad[mono_new]        
            image_gauche = self.objetgauche - self.rad[mono_new]
            if image_gauche[0] < self.pos[mono_new][0] < image_droit[0] and image_gauche[1] < self.pos[mono_new][1] < image_droit[1]:   #we initialaze the position and we want no particles in our object
                NoOverlap = False 
            while k < mono_new and NoOverlap :
                dx = self.pos[mono_new][0] - self.pos[k][0]
                dy = self.pos[mono_new][1] - self.pos[k][1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance < radius_mono_new + self.rad[k] :
                    NoOverlap = False
                else:
                    k += 1    
            if NoOverlap :
                mono_new += 1

            #-->> your turn
            #Place one monomer after another, such that
            #initial configuration has random positions without
            #overlaps between monomers and with confining hard walls.
            
        if mono_new != self.NM:
            print('Failed to initialize all particle positions.\nIncrease simulation box size!')
            exit()
        
    
    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
        
    def Wall_time(self):
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''
        
        #---> Your turn!
        # compute list with all collision times with left and right wall
        x_coll = np.where(self.vel[:,0]<0, self.BoxLimMin[0] + self.rad, self.BoxLimMax[0] - self.rad)   #we have two types of wall collision : with the box and with our object
        List_dt_x =  (x_coll - self.pos[:,0] ) / self.vel[:,0] 
        x_coll_objet = np.zeros((self.NM))
        for k in range(self.NM):                                  #this programme is explained in the report 
            if self.objetgauche[1] - self.rad[k] < self.pos[k,1] < self.objetdroit[1] + self.rad[k] :
                if self.vel[k,0]>0 :
                    if self.objetgauche[0] - self.rad[k] < self.pos[k,0]:
                        x_coll_objet[k] = np.inf
                    else :
                        x_coll_objet[k] = self.objetgauche[0] - self.rad[k]
                else :
                    if self.objetdroit[0] + self.rad[k] > self.pos[k,0]:
                        x_coll_objet[k] = np.inf
                    else :
                        x_coll_objet[k] = self.objetdroit[0] + self.rad[k]
            else :
                x_coll_objet[k] = np.inf

        List_dt_x_obj = abs ((x_coll_objet - self.pos[:,0] ) / self.vel[:,0])

        y_coll = np.where(self.vel[:,1]<0, self.BoxLimMin[1] + self.rad, self.BoxLimMax[1] - self.rad) 
        List_dt_y =  (y_coll - self.pos[:,1] ) / self.vel[:,1]

        y_coll_objet = np.zeros((self.NM))
        for k in range(self.NM):
            if self.objetgauche[0] - self.rad[k] < self.pos[k,0] < self.objetdroit[0] + self.rad[k] :
                if self.vel[k,1]>0 :
                    if self.objetgauche[1] - self.rad[k] < self.pos[k,1]:
                        y_coll_objet[k] = np.inf
                    else :
                        y_coll_objet[k] = self.objetgauche[1] - self.rad[k]
                else :
                    if self.objetdroit[1] + self.rad[k] > self.pos[k,1]:
                        y_coll_objet[k] = np.inf
                    else :
                        y_coll_objet[k] = self.objetdroit[1] + self.rad[k]
            else :
                y_coll_objet[k] = np.inf
                
 
        List_dt_y_obj = abs ((y_coll_objet - self.pos[:,1] ) / self.vel[:,1])

        # search for the index of the minimum collision time 
        index_MinTimex = np.argmin(  List_dt_x )
        index_MinTimey = np.argmin(  List_dt_y )
        index_MinTimexobj = np.argmin(  List_dt_x_obj )
        index_MinTimeyobj = np.argmin(  List_dt_y_obj ) 

        list_smallestind = [List_dt_x[index_MinTimex], List_dt_x_obj[index_MinTimexobj], List_dt_y[index_MinTimey], List_dt_y_obj[index_MinTimeyobj]]
        index_Min = np.argmin(list_smallestind)

        
        #the indice of the minimum time is know so now we can definie wether it is a horzontal or vertical collision, and the time left before collision
        if index_Min <= 1 :
            wall_direction = 0
            if index_Min == 0 :
                minCollTime = List_dt_x[ index_MinTimex ]
                collision_disk = index_MinTimex
            else :
                minCollTime = List_dt_x_obj[ index_MinTimexobj ]
                collision_disk = index_MinTimexobj
            
        else :
            wall_direction = 1
            if index_Min ==2 :
                minCollTime = List_dt_y[ index_MinTimey ]
                collision_disk = index_MinTimey
            else :
                minCollTime = List_dt_y_obj[ index_MinTimeyobj ]
                collision_disk = index_MinTimeyobj
            
        
        
        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.mono_1 = collision_disk
        self.next_wall_coll.w_dir = wall_direction
        
        
    def Mono_pair_time(self):
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''
        mono_i = self.mono_pairs[:,0] # List of collision partner 1
        mono_j = self.mono_pairs[:,1] # List of collision partner 2
     
        # Your turn!
        dx_dy_ofPairs = self.pos[mono_i] - self.pos[mono_j]
        dv_ofPairs = self.vel[mono_i] - self.vel[mono_j]
        
        a = dv_ofPairs[:,0]**2 + dv_ofPairs[:,1]**2
        b = 2 * (dv_ofPairs[:,0]*dx_dy_ofPairs[:,0] + dv_ofPairs[:,1]*dx_dy_ofPairs[:,1])
        c = dx_dy_ofPairs[:,0]**2 + dx_dy_ofPairs[:,1]**2 - (self.rad[mono_j] + self.rad[mono_i])**2
        
        omega = b**2 - 4*a*c
    
        
        #CollTime = np.zeros(len(self.mono_pairs[:,0]))
        #for k in range(len(self.mono_pairs[:,0])) :
        #print(k)
        #if omega[k] > 0 and b[k]<0:
        #CollTime[k] = (1/2*a[k]) * (-b[k] - np.sqrt(omega[k]))
        #else :
        #CollTime[k] = np.inf
        #print('CollTime= ', CollTime)


        CollTime=np.where((omega>0) & (b<0),(1/(2*a)) * (-b-np.sqrt(omega)),np.inf)

        PairOfMono = np.argmin(CollTime)
        minCollTime = CollTime[PairOfMono]
        
        #print("dx_dy_ofPairs = ", dx_dy_ofPairs)
        #dist_ofPairs = np.linalg.norm(dx_dy_ofPairs, axis=1 )
        #norm_vel = np.linalg.norm(self.vel, axis=1)
        
        #List_tcoll = np.empty(...)
        #for k in range(len(dist_ofPairs)[0]) :
        #(i,j) = np.arg(self.mono_pairs)
        #sum_vel = norm_vel[i] + norm_vel[j]
        #List_tcoll[k] = dist_ofPairs[k]/sum_vel
        
        #pairIndex_minDist = np.argmin(dist_ofPairs)
        #mono_i, mono_j = self.mon_pairs[ pairIndex_minDist ]
        #minDist = dist_ofPairs[ pairIndex_minDist ]
        
        #minCollTime = np.inf #This is a dummy. Write the code!
        collision_disk_1 = self.mono_pairs[PairOfMono,0] #This is a dummy. Write the code!
        collision_disk_2 = self.mono_pairs[PairOfMono,1] #This is a dummy. Write the code!

        self.next_mono_coll.dt = minCollTime
        self.next_mono_coll.mono_1 = collision_disk_1
        self.next_mono_coll.mono_2 = collision_disk_2
        #self.next_mono_coll.w_dir = not necessary
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
        
        #This is not correct! you have to write the code!
        
        self.Wall_time()
        self.Mono_pair_time()
        
        if  self.next_wall_coll.dt < self.next_mono_coll.dt:     #choosing what the next event will be

            return self.next_wall_coll
        else :
            return self.next_mono_coll
        
        
            
    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''
        
        #write the code!
        
        if next_event.Type == 'wall' :
            mono_index = next_event.mono_1
            wall_direction = next_event.w_dir
            self.vel[ mono_index , wall_direction ] *= -1
        
        else :
            mono_1 = next_event.mono_1
            mono_2 = next_event.mono_2
            
            del_pos = self.pos[mono_2] - self.pos[mono_1]
            del_pos/=np.linalg.norm(del_pos)
            del_vel = self.vel[mono_1] - self.vel[mono_2]
            
            M1 = self.mass[mono_1]
            M2 = self.mass[mono_2]
            self.vel[mono_1] -=((2*M2)/(M1 + M2)) * np.dot(del_pos,del_vel) * del_pos
            self.vel[mono_2] +=((2*M1)/(M1 + M2)) * np.dot(del_pos,del_vel) * del_pos
            

    def compute_new_color(self, next_event, time):
        if next_event.Type == 'mono' :
            mono_1 = next_event.mono_1
            mono_2 = next_event.mono_2
            proba = np.random.random()    #proba is a random number between 0 and 1
            if proba > 0.2 :             #when a healthy particle meets a sick particle, it has a probability defined here of becoming sick
                if self.color[mono_1] == 0.2 and self.color[mono_2] ==0.8:
                    self.color[mono_2] = 0.2
                    self.time_recover[mono_2] = time + 3
                if self.color[mono_1] == 0.8 and self.color[mono_2] ==0.2:
                    self.color[mono_1] = 0.2
                    self.time_recover[mono_1] = time + 3


    def compute_recover(self, indice):
        self.color[indice] = 0.4  #color of the recovered particles 
            
        
        

    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):     
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        Lr_xMin, Lr_xMax = 4, 6 #size of our object 
        Lr_yMin, Lr_yMax = 4, 16 #size of our object 
        BorderGap = 0.1*(L_xMax - L_xMin)
        BorderGap2 = 0.1*(Lr_xMax - Lr_xMin)  #we plot our object thanks to the same code as the one for the box
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
        

        ax.set_xlim(Lr_xMin-BorderGap2, Lr_xMax+BorderGap2) 
        ax.set_ylim(Lr_yMin-BorderGap2, Lr_yMax+BorderGap2) 
        
        #--->plot hard walls (rectangle)
  
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin,linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        MonomerColors = self.color
        MonomerColors = np.linspace( 0.2, 0.2, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig(FileName)
        plt.close()

        
class Dimers(Monomers):
    """
    --> Class derived from Monomers.
    --> See also comments in Monomer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with DIMERS (and monomers). Two hard-sphere monomers form a dimer,
    and experience additional ellastic collisions at the maximum
    bond length of the dimer. The bond length is defined in units of the
    minimal distance of the monomers, i.e. the sum of their radiai.
    -Next to the monomer information, the maximum dimer bond length is needed
    to fully describe one configuration.
    -Initial configuration of $N$ monomers has random positions without overlap
    and separation of dimer pairs is smaller than the bond length.
    Velocities have random orientations and norms that satisfy
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for all inter-particle collsions is the mono_pair array
    (explained in momonmer class). Essentail for the ellastic bond collision
    of the dimers is the dimer_pair array which book-keeps index pairs of
    monomers that form a dimer. For example, for a system of $N = 10$ monomers
    and $M = 2$ dimers:
    monomer indices = 0, 1, 2, 3, ..., 9
    dimer_pair = [[0,2], [1,3]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2, mono_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers mono_4,..., mono_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (mono_0, mono_2), (mono_1, mono_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfDimers = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), bond_length_scale = 1.2, k_BT = 1):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        assert ( (NumberOfDimers > 0) and (NumberOfMonomers >= 2*NumberOfDimers) )
        assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
        Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
        self.ND = NumberOfDimers
        self.dimer_pairs = np.array([[k,self.ND+k] for k in range(self.ND)])#choice 2 -> more practical than [2*k,2*k+1]
        mono_i, mono_j = self.dimer_pairs[:,0], self.dimer_pairs[:,1]
        self.bond_length = bond_length_scale * ( self.rad[mono_i] + self.rad[mono_j] )
        self.next_dimer_coll = CollisionEvent( Type = 'dimer', dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)
        
        '''
        Positions initialized as pure monomer system by monomer __init__.
        ---> Reinitalize all monomer positions, but place dimer pairs first
        while respecting the maximal distance given by the bond length!
        '''
        self.assignRandomDimerPos()
        self.assignRandomMonoPos( 2*NumberOfDimers )
    
    def assignRandomDimerPos(self):
        '''
        Make this is a PRIVATE function -> cannot be called outside class definition
        initialize random positions without overlap between monomers and wall
        '''
        dimer_new_index, infiniteLoopTest = 0, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while dimer_new_index < self.ND and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            mono_i, mono_j = dimer_new = self.dimer_pairs[dimer_new_index]
            self.pos = np.array( [[ 1.5,  1.5],
                              [ 4.79516011, 3.19684253],
                              [ 2.49474337,  3.46710566],
                              [ 5.74852769,  2.46821118]] )
            dimer_new_index += 1
            # Your turn to place the dimers one after another such that
            # there are no overlaps between monomers and hard walls and
            # dimer pairs are not further appart than their max bond length.
        if dimer_new_index != self.ND:
            print('Failed to initialize all dimer positions.\nIncrease simulation box size!')
            exit()
        
        
    def __str__(self, index = 'all'):
        if index == 'all':
            return Monomers.__str__(self) + "\ndimer pairs = " + str(self.dimer_pairs) + "\nwith max bond length = " + str(self.bond_length)
        else:
            return "\nDimer pair " + str(index) + " consists of monomers = " + str(self.dimer_pairs[index]) + "\nwith max bond length = " + str(self.bond_length[index]) + Monomers.__str__(self, self.dimer_pairs[index][0]) + Monomers.__str__(self, self.dimer_pairs[index][1])

    def Dimer_pair_time(self):
        '''
        Function computes list of remaining time dt until
        future dimer bond collition for all dimer pairs.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_dimer_coll.
        '''
        mono_i = self.dimer_pairs[:,0] # List of collision partner 1
        mono_j = self.dimer_pairs[:,1] # List of collision partner 2

        
        dx_dy_ofPairs = self.pos[mono_i] - self.pos[mono_j]
        dv_ofPairs = self.vel[mono_i] - self.vel[mono_j]
        
        a = dv_ofPairs[:,0]**2 + dv_ofPairs[:,1]**2
        b = 2 * (dv_ofPairs[:,0]*dx_dy_ofPairs[:,0] + dv_ofPairs[:,1]*dx_dy_ofPairs[:,1])
            #if self.rad[mono_j] > self.rad[mono_i] :
            #max_rad = self.rad[mono_j]
            #else :
            #max_rad = self.rad[mono_i]
        c = dx_dy_ofPairs[:,0]**2 + dx_dy_ofPairs[:,1]**2 - (self.bond_length)**2
        
        omega = b**2 - 4*a*c
        
        CollTime=np.where((omega>0),(1/(2*a)) * (-b + np.sqrt(omega)),np.inf)
        
        PairOfMono = np.argmin(CollTime)
        minCollTime = CollTime[PairOfMono]
        
        
        collision_disk_1 = self.dimer_pairs[PairOfMono,0] 
        collision_disk_2 = self.dimer_pairs[PairOfMono,1]
        
        self.next_dimer_coll.dt = minCollTime
        self.next_dimer_coll.mono_1 = collision_disk_1
        self.next_dimer_coll.mono_2 = collision_disk_2
        #self.next_dimer_coll.w_dir = not necessary
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
        
        self.Wall_time()
        self.Mono_pair_time()
        self.Dimer_pair_time()
        
        if  self.next_wall_coll.dt < self.next_mono_coll.dt and self.next_wall_coll.dt < self.next_dimer_coll.dt :
            return self.next_wall_coll
        elif self.next_mono_coll.dt < self.next_dimer_coll.dt:
            return self.next_mono_coll
        else :
            return self.next_dimer_coll
        

    
        
    def snapshot(self, FileName = './snapshot.png', Title = ''):
        '''
        ---> Overwriting snapshot(...) of Monomers class!
        Function saves a snapshot of current configuration,
        i.e. monomer positions as circles of corresponding radius,
        dimer bond length as back empty circles (on top of monomers)
        velocities as arrows on monomers,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        COLORS = np.linspace(0.2,0.95,self.ND+1)
        MonomerColors = np.ones(self.NM)*COLORS[-1] #unique color for monomers
        # recolor each monomer pair with individual color
        MonomerColors[self.dimer_pairs[:,0]] = COLORS[:len(self.dimer_pairs)]
        MonomerColors[self.dimer_pairs[:,1]] = COLORS[:len(self.dimer_pairs)]

        #plot solid monomers
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)
        
        #plot bond length of dimers as black cicles
        Width, Hight, Angle = self.bond_length, self.bond_length, np.zeros( self.ND )
        mono_i = self.dimer_pairs[:,0]
        mono_j = self.dimer_pairs[:,1]
        collection_mono_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_i],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        collection_mono_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_j],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        ax.add_collection(collection_mono_i)
        ax.add_collection(collection_mono_j)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig( FileName)
        plt.close()
