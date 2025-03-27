""" Processing Center for Lab 4 data

Returns:
         A graphical display of the given data set's 
         linear trend for Flexure Modulus analysis,
         and a data file containing the results for
         stress and strain at break along with the 
         modulus and R-squared value of the fit.
"""
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os 
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# Name of the output file
Data_File = 'Data_File.csv'
# Read the CSV file
folder = "Lab 4 Data"
folder = os.listdir(folder)
# Define the linear function for the best fit
def fcn(x, a, b):
    return a*x + b
# Set up the figure for plotting
fig, axs = plt.subplots(3, 4, figsize=(15, 10))
# Flatten the 2D array of axes for easier indexing
axs = axs.ravel()  
plot_index = 0
#print(folder)
###
# Cycle through the files in the folder
# and process the data
i=0 
for file in folder:
    i+=1
    if file.endswith('.csv'):
        # print(f'{i}:  {file}')
        f1 = 'Lab 4 Data/' + file
        data = pd.read_csv(f1, header=6)
        
        Time = data['(sec)']
        Exts = data['(mm)']
        Load = data['(N)']
        ###################
        # Normilize the data!
        Load = (Load - Load[0])*(-1)
        Exts = (Exts - Exts[0])*(-1)
        ###
        # Dimensional Constants
        dim = pd.read_csv(f1, nrows=5,header=0)

        #print(dim.head())
        L = dim['Text Inputs : Specimen label'][1]
        d = dim['Text Inputs : Specimen label'][2]
        b = dim['Text Inputs : Specimen label'][3]
        L = np.float64(L)
        d = np.float64(d)
        b = np.float64(b)
        print(L,d,b)
        # print(data.head())
        # print(data.shape)
        # print(data.columns)
        # print(data.dtypes)
        # print(data.info())
        # print(data.describe())
        
        # print(Time.head())
        # print(Exts.head())
        # print(Load.head())
        ###
        # Define rounding for the maximum value
        sigfig=0
        MxApp = np.round(np.max(Load),decimals=sigfig)
        ###
        # Peak Detection
        for ii in range(1,len(Load)):
            # If maximum reached:
            if np.round(Load[ii],decimals=sigfig) == MxApp:
                # Ensure the index
                if Load[ii+1] < Load[ii]:
                    ###
                    # set maximum & index
                    Mx   = Load[ii]
                    maxi = ii
                    ###
                    # assumed based off first plot
                    g0 = np.array([0.0, 0.0])
                    # Call best fit function
                    popt, pcov = curve_fit(fcn, Time[:maxi], Load[:maxi], p0=g0)
                    # extract error from covariance matrix
                    err = np.sqrt(np.diag(pcov))
                    ###
                    # calcualte the coefficiant of determination
                    observed = Load[:maxi]
                    expected = fcn(Time[:maxi], *popt)
                    r2 = r2_score(observed, expected)
                    ###
                    ###
                    ################
                    # Calculations                    
                    Stress = (3*Load*L)/(2*b*(d**2))
                    Strain = (6*Exts*d)/(L**2)
                    ###
                    # Index at break
                    Stress_at_brk = Stress[maxi]
                    Strain_at_brk = Strain[maxi]
                    ###
                    # Print to user
                    out = f""" 
-----------------------------
| File:       {file}
| Max: {Mx}  Index: {maxi} {ii}
-----------------------------
| Fit:        {popt[0]:.2f}x + {popt[1]:.2f} 
| Err:(+/-)   {err[0]:.2f}x  + {err[1]:.2f}
| R-squared:  {r2}
-----------------------------
| --- At Break ---
| Stress:     {Stress_at_brk}
| Strain:     {Strain_at_brk}
-----------------------------
| Dimensions
| L:          {L}
| d:          {d}
| b:          {b}
-----------------------------
                    """
                    print(out)
                    ###
                    # Set plot
                    axs[plot_index].plot(Time[:maxi], Load[:maxi], 'b-', label=f'D{ii}')
                    axs[plot_index].plot(Time[:maxi], fcn(Time[:maxi], *popt), 'r-', label=f'fit{ii}')
                    axs[plot_index].legend()
                    axs[plot_index].set_xlabel('Time (s)')
                    axs[plot_index].set_ylabel('Load (N)')
                    axs[plot_index].set_title(f"Data: {file}")
                    plot_index += 1
                    ###
                    # Create a dictionary to store the data
                    data_dict = {
                        'File': [file],
                        'Modulous': [popt[0]],
                        'Stress': [Stress_at_brk],
                        'Strain': [Strain_at_brk],
                        'Rsqr': [r2],
                    }
                    # Convert to DataFrame
                    df = pd.DataFrame(data_dict)
                    # Save to CSV - mode='a' for append, header=False if file exists
                    if not os.path.exists(Data_File):
                        df.to_csv(Data_File,  mode='w')
                    else:
                        df.to_csv(Data_File,  mode='a', header=False)
                    break
###
# Display the plots
# to user
plt.show()  