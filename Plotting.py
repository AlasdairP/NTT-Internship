import matplotlib.pyplot as plt
import numpy as np
import math

def Probability_density(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid):
    
    if state_index == 0:
        label = "1s"
    elif state_index == 1:
        label = "2p"
    elif state_index == 2:
        label = "2p"
    elif state_index == 3:
        label = "2s"
    
    Nx = len(XX_Ang)
    Ny = len(YY_Ang)
    
    state = states[state_index]
    
    # "unravel" - initially thought (Nx,Ny), but this works
    density = (np.absolute(state)**2).reshape((Ny,Nx))  
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    c = ax.pcolormesh(XX_Ang,YY_Ang, density, shading='auto')
    if Electric:
        if Magnetic:
            ax.set_title(str(label) + ", E=" + str(F_kV) + "kV/cm, B=" + str(B_Tesla[i]) + "T, " + grid)
        else:
            ax.set_title(str(label) + ", E=" + str(F_kV[i]) + "kV/cm, B=0, " + grid)
    elif Magnetic:
        ax.set_title(str(label) + ", E=0"  ", B=" + str(B_Tesla[i]) + "T, " + grid)
    else:
        ax.set_title(str(label) + ", no fields, " + str(grid))
    ax.set_xlabel("x (Ang.)")
    ax.set_ylabel("y (Ang.)")
    fig.colorbar(c,ax=ax)
    fig.tight_layout()
    return fig

def Phase(states,state_index,XX_Ang,YY_Ang,Electric,Magnetic,F_kV,B_Tesla,i,grid):
    
    if state_index == 0:
        label = "1s"
    elif state_index == 1:
        label = "2p"
    elif state_index == 2:
        label = "2p"
    elif state_index == 3:
        label = "2s"
    
    Nx = len(XX_Ang)
    Ny = len(YY_Ang)
    
    state = states[state_index]
    state = state.reshape((Ny,Nx)) 
    arg = np.angle(state)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    c = ax.pcolormesh(XX_Ang,YY_Ang, arg, vmin=0, vmax = 2*math.pi, shading='auto')
    if Electric:
        if Magnetic:
            ax.set_title(str(label) + ", E=" + str(F_kV) + "kV/cm, B=" + str(B_Tesla[i]) + "T, " + grid)
        else:
            ax.set_title(str(label) + ", E=" + str(F_kV[i]) + "kV/cm, B=0, " + grid)
    elif Magnetic:
        ax.set_title(str(label) + ", E=0"  ", B=" + str(B_Tesla[i]) + "T, " + grid)
    else:
        ax.set_title(str(label) + ", no fields, " + str(grid))    
    fig.colorbar(c,ax=ax)
    fig.tight_layout()
    return fig
    
def Energy_vs_field(energy_lists,k,Electric,Magnetic,F_kV,B_Tesla,grid):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if Electric:
        if Magnetic:    
            for state in range(k):
            # Adjusted to show shifts all starting from zero
                if state == 0:
                    label = "1s"
                elif state == 1:
                    label = "2py"
                elif state == 2:
                    label = "2px"
                elif state == 3:
                    label = "2s"
                ax.plot(B_Tesla,(energy_lists[state]-energy_lists[state][0]), 'o',label=label)
                #ax.plot(B_Tesla,energy_lists[state], 'o')     # original, true energies, not deltaE
            ax.set_title("Magnetic field effect, with E=" + str(F_kV) + ", " + grid)
            ax.set_xlabel("B (T)")
        else:
            for state in range(k):
                # Adjusted to show shifts all starting from zero
                if state == 0:
                    label = "1s"
                elif state == 1:
                    label = "2px"
                elif state == 2:
                    label = "2py"  # note 2px and 2py swap if E=Ex since 2px has higher polarisability and thus larger negative shift and becomes the 2nd state in ascending order
                elif state == 3:
                    label = "2s"
                ax.plot(F_kV,(energy_lists[state]-energy_lists[state][0]),'o',label=label)
                #ax.plot(F_kV,energy_lists[state], 'o')
            ax.set_title("E field effect (Stark), with B=0, " + grid)
            ax.set_xlabel("E (kV/cm)")
    elif Magnetic:
        for state in range(k):
            # Adjusted to show shifts all starting from zero
            if state == 0:
                label = "1s"
            elif state == 1:
                label = "2py"
            elif state == 2:
                label = "2px"
            elif state == 3:
                label = "2s"
            ax.plot(B_Tesla,(energy_lists[state]-energy_lists[state][0]), 'o',label=label)
        ax.set_title("Magnetic field effect, with E=0, " + grid)
        ax.set_xlabel("B (T)")
    else:
        return 
    ax.set_ylabel("Change in binding energy (meV)")
    ax.legend()
    fig.tight_layout()
    return fig 
    

def Potential_linecut(x_Ang,V_plot,y_cut):
    
    Nx = len(x_Ang)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_Ang, V_plot[y_cut])
    ax.set_title("Linecut of potential at y index " + str(y_cut) + " / " + str(Nx))
    ax.set_xlabel("x (Ang)")
    ax.set_ylabel("Potential")
    fig.tight_layout()

    return fig
    
    
    
    
"""
# Plot linecut of wavefunction
y_cut = 40
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot((x*Bohr_to_Ang), density[y_cut])
ax.set_title("Linecut of prob. density at y index " + str(y_cut) + " / " + str(Ny))
ax.set_xlabel("x (Ang)")
ax.set_ylabel("Prob. density")
fig.tight_layout()
plt.show()
"""


"""
# Plot energy against state index
index = np.arange(1,k+1)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(index,energies,'o')
ax.set_title("Non-hydrogenic Rydberg series")
ax.set_xlabel("State index n")
ax.set_ylabel("Binding energy (meV)")
fig.tight_layout()
plt.show()
"""


"""
### Plot colour map of just x coordinates, to see exp scaling
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
c = ax.pcolormesh(XX)
fig.colorbar(c,ax=ax)
plt.show()
"""


"""
        ########### Plot phase - in loop #############
        
        state = states[k-1].reshape((Ny,Nx)) 
        arg = np.angle(state)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        c = ax.pcolormesh(XX_Ang,YY_Ang, arg, vmin=0, vmax = 2*math.pi, shading='auto')
        ax.set_title("2py Phase, B = " + str(B_Tesla[i]) + "T, " + grid)
        fig.colorbar(c,ax=ax)
        fig.tight_layout()
        plt.show()
        
        if i == 0:
            original_density = density
        # Change from original
        else:
            change = density - original_density
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            c = ax.pcolormesh(XX_Ang,YY_Ang, change, shading='auto')
            ax.set_title("2s, B = " + str(B_Tesla[i]) + "T. Change from B = 0 state")
            ax.set_xlabel("x (Ang.)")
            ax.set_ylabel("y (Ang.)")
            fig.colorbar(c,ax=ax)
            fig.tight_layout()
            plt.show()
"""