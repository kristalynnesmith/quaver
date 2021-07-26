######
######
import os
import http
from astropy.coordinates.name_resolve import NameResolveError
#########
##########

import lightkurve as lk
import numpy as np

#################
#################
import matplotlib
import tkinter
matplotlib.use("TkAgg")
####################
###################

import matplotlib.pyplot as plt
from astropy.coordinates import get_icrs_coordinates
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy.wcs import *
from astropy import units as u
import astropy.io.fits as pyfits
from copy import deepcopy
from matplotlib import gridspec
from matplotlib import patches
import sys

############################################
#Define function to record the positions of clicks in the pixel array image for the extraction mask.
def onclick(event):

    global ix,iy
    ix,iy = int(round(event.xdata)),int(round(event.ydata))

    global coords
    coords.append((ix,iy))

    global row_col_coords
    row_col_coords.append((iy,ix))

    #plt.plot(event.xdata,event.ydata,marker=u"$\u2713$",color='limegreen',markersize=9)
    plt.plot(ix,iy,marker=u"$\u2713$",color='red',markersize=9)
    fig.canvas.draw()

    print(ix,iy)
#############################################


############################################
#Define function to record the X-positions of the cadences to mask out if needed.
def onclick_cm(event):

    global ix_cm
    ix_cm = int(round(event.xdata))

    global masked_cadence_limits
    masked_cadence_limits.append(ix_cm)
    print(ix_cm)

        #plt.axvspan(masked_cadence_limits[0],masked_cadence_limits[1],color='r')
    plt.axvline(x=ix_cm,color='red')
    fig_cm.canvas.draw()

   # print('Masking cadences '+str(masked_cadence_limits[0])+" --> "+str(masked_cadence_limits[1]))


#############################################
#############################################

#Set the dimension of the downloaded TPF (currently only works for squares):

tpf_width_height = 26

#Define target and obtain DSS image from coordinates.

#target = '2MASX J11580219+5920538'
try :
    target = input('Target Common Name: ')
    target_coordinates = target
    source_coordinates = get_icrs_coordinates(target)       #this requires that SIMBAD be up and working...
    print(source_coordinates)
    print("\n")

# If target is not found by name use Sky Coordinates
# Enter as glactic coordinates for simple reference to SIMBAD

except NameResolveError:
    print("\n"+"Could not find target by name provided. Try Sky Coordinates.\n")
    # print("Input as Galactic Coordinates: l, b  (in deg)")
    print("Input as ICRS: RA, Dec  (in deg)")
    print("Use full precision from NED for best results.\n")

    # long = float(input('Longitude: '))
    # lat = float(input('Latitude: '))
    ra = float(input('RA: '))
    dec = float(input('Dec: '))

    # gal_coordinates = SkyCoord(long, lat ,frame='galactic',unit='deg')
    # source_coordinates = gal_coordinates.transform_to('icrs')
    source_coordinates = SkyCoord(ra,dec,frame='icrs',unit='deg')
    # source_coordinates = SkyCoord(ra,dec,frame='icrs',unit=(u.hourangle, u.deg))

    target = input('Target Common Name: ')
    target_coordinates = str(ra)+" "+str(dec)

    print(source_coordinates)
    print("\n")
#############################################
#############################################


dss_image = SkyView.get_images(position=source_coordinates,survey='DSS',pixels=str(400))
wcs_dss = WCS(dss_image[0][0].header)
dss_pixmin = np.min(dss_image[0][0].data)
dss_pixmax = np.max(dss_image[0][0].data)
dss_pixmean = np.mean(dss_image[0][0].data)

dss_head = dss_image[0][0].header
dss_ra = dss_head['CRVAL1']
dss_dec = dss_head['CRVAL2']


#Retrieve the available tesscut data for FFI-only targets.
sector_data = lk.search_tesscut(target_coordinates)
num_obs_sectors = len(sector_data)

if num_obs_sectors == 0:
    print("This object has not been observed by TESS.")

    sys.exit()

print(sector_data)

#Set cycle of interest:

cycle = int(input('Enter Cycle: '))


first_sectors = [1,14,27,40]
last_sectors = [13,26,39,55]

if cycle == 1:
    first_sector = first_sectors[0]
    last_sector = last_sectors[0]
elif cycle ==2:
    first_sector = first_sectors[1]
    last_sector = last_sectors[1]
elif cycle ==3:
    first_sector = first_sectors[2]
    last_sector = last_sectors[2]
elif cycle==4:
    first_sector = first_sectors[3]
    last_sector = last_sectors[2]
else:
    print('Invalid Cycle Number')


unstitched_lc_regression = []
unstitched_lc_pca = []

if num_obs_sectors == 0:
    print("This object has not been observed by TESS.")

    sys.exit()

else:

    for i in range(0,num_obs_sectors):

        try:


            tpf = sector_data[i].download(cutout_size=(tpf_width_height, tpf_width_height)) #gets earliest sector

            sector_number = tpf.get_header()['SECTOR']
            sec = str(sector_number)
            ccd = tpf.get_header()['CCD']
            cam = tpf.get_header()['CAMERA']

            if sector_number >= first_sector and sector_number <= last_sector:  #Check that this TPF is in the desired Cycle/year

                print("Generating pixel map for sector "+sec+".\n")

                #Check that this object is actually on silicon and getting data (not always the case just because TESSCut says so).
                #By making a light curve from a dummy aperture of the middle 5x5 square and seeing if its mean flux is zero.

                aper_dummy = np.zeros(tpf[0].shape[1:], dtype=bool) #blank
                aper_dummy[int(tpf_width_height/2-3):int(tpf_width_height/2+3),int(tpf_width_height/2-3):int(tpf_width_height/2+3)] = True
                lc_dummy = tpf.to_lightcurve(aperture_mask=aper_dummy)

                # If there is no observation in this sector, there most likely won't be one in the next.
                # No need to check through the rest of the sectors in this cycle.


                if np.mean(lc_dummy.flux) != 0:

                    hdu = tpf.get_header(ext=2)

                    #Get WCS information and flux stats of the TPF image.
                    tpf_wcs = WCS(tpf.get_header(ext=2))

                    pixmin = np.min(tpf.flux[200])
                    pixmax = np.max(tpf.flux[200])
                    pixmean = np.mean(tpf.flux[200])

                    temp_min = float(pixmin)
                    # print(temp_min)
                    temp_max = float(1e-3*pixmax+pixmean)
                    # print(temp_max)

                    #Create a blank boolean array for the aperture, which will turn to TRUE when pixels are selected.

                    aper = np.zeros(tpf[0].shape[1:], dtype=bool) #blank
                    aper_mod = aper.copy()       #For the source aperture
                    aper_buffer = aper.copy()    #For the source aperture plus a buffer region to exclude from both additive and mult. regressors

                    aper_width = tpf[0].shape[1]
                    #Plot the TPF image and the DSS contours together, to help with aperture selection, along with the starter aperture.

                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(111,projection=tpf_wcs)
                    # ax.imshow(tpf.flux[200],vmin=pixmin,vmax=1e-3*pixmax+pixmean)
                    ax.imshow(tpf.flux[200],vmin=temp_min,vmax=temp_max)
                    ax.contour(dss_image[0][0].data,transform=ax.get_transform(wcs_dss),levels=[0.4*dss_pixmax,0.5*dss_pixmax,0.75*dss_pixmax],colors='white',alpha=0.9)
                    ax.scatter(aper_width/2.0,aper_width/2.0,marker='x',color='k',s=8)

                    ax.set_xlim(-0.5,aper_width-0.5)  #This section is needed to fix the stupid plotting issue in Python 3.
                    ax.set_ylim(-0.5,aper_width-0.5)

                    plt.title('Define extraction pixels:')
                    coords = []
                    row_col_coords = []
                    cid = fig.canvas.mpl_connect('button_press_event',onclick)

                    plt.show()
                    plt.close(fig)

                    fig.canvas.mpl_disconnect(cid)

                    for i in range(0,len(row_col_coords)):

                        aper_mod[row_col_coords[i]] = True

                        min_coord_index = np.min(row_col_coords)
                        max_coord_index = np.max(row_col_coords)
                        aper_buffer[min_coord_index:max_coord_index,min_coord_index:max_coord_index] = True


                    #Create a mask that finds all of the bright, source-containing regions of the TPF.
                    #Need to change to prevent requiring contiguous mask:
                    '''
                    thumb = np.nanpercentile(tpf.flux, 95, axis=0)
                    thumb -= np.nanpercentile(thumb, 20)
                    allbright_mask = thumb > np.percentile(thumb, 40)
                    '''
                    allbright_mask = tpf.create_threshold_mask(threshold=1.5,reference_pixel=None)
                    allfaint_mask = ~allbright_mask

                    allbright_mask &= ~aper_buffer
                    allfaint_mask &= ~aper_buffer

                    #New attempt to get the additive background first:

                    additive_bkg = lk.DesignMatrix(tpf.flux[:, allfaint_mask]).pca(3)
                    additive_bkg_and_constant = additive_bkg.append_constant()

                    #Add a module to catch possible major systematics that need to be masked out before continuuing:

                    max_masked_regions = 3 #set maximum number of regions of the light curve that can be masked out.
                    number_masked_regions = 1 #set to 1 at first

                    if np.abs(np.max(additive_bkg.values)) > 0.15:   #None of the normally extracted objects has additive components with absolute values over 0.2 ish.

                        redo_with_mask = input('Additive components indicate major systematics; add a cadence mask (Y/N) ?')

                        if redo_with_mask == 'Y' or redo_with_mask=='y' or redo_with_mask=='YES' or redo_with_mask=='yes':


                            fig_cm = plt.figure()
                            ax_cm = fig_cm.add_subplot()
                            ax_cm.plot(additive_bkg.values)

                            plt.title('Select first and last cadence to define mask region:')
                            masked_cadence_limits = []
                            cid_cm = fig_cm.canvas.mpl_connect('button_press_event',onclick_cm)

                            plt.show()
                            plt.close(fig_cm)

                            if masked_cadence_limits[0] >= 0:
                                first_timestamp = tpf.time[masked_cadence_limits[0]]
                            else:
                                first_timestamp = 0
                            if masked_cadence_limits[1] < len(tpf.time) -1:
                                last_timestamp = tpf.time[masked_cadence_limits[1]]
                            else:
                                last_timestamp = tpf.time[len(tpf.time)-1]

                            cadence_mask = ~((tpf.time > first_timestamp) & (tpf.time < last_timestamp))

                            tpf = tpf[cadence_mask]

                            additive_bkg = lk.DesignMatrix(tpf.flux[:, allfaint_mask]).pca(3)
                            additive_bkg_and_constant = additive_bkg.append_constant()

                            print(np.abs(np.max(additive_bkg.values)))


                            for i in range(0,max_masked_regions):


                                if np.abs(np.max(additive_bkg.values)) > 0.2  and number_masked_regions <= max_masked_regions:

                                    number_masked_regions += 1

                                    print('Systematics remain; define the next masked region.')
                                    print(np.max(additive_bkg.values))
                                    fig_cm = plt.figure()
                                    ax_cm = fig_cm.add_subplot()
                                    ax_cm.plot(additive_bkg.values)

                                    plt.title('Select first and last cadence to define mask region:')
                                    masked_cadence_limits = []
                                    cid_cm = fig_cm.canvas.mpl_connect('button_press_event',onclick_cm)

                                    plt.show()
                                    plt.close(fig_cm)


                                    if masked_cadence_limits[0] >= 0:
                                        first_timestamp = tpf.time[masked_cadence_limits[0]]
                                    else:
                                        first_timestamp = 0
                                    if masked_cadence_limits[1] < len(tpf.time) -1:
                                        last_timestamp = tpf.time[masked_cadence_limits[1]]
                                    else:
                                        last_timestamp = tpf.time[-1]


                                    cadence_mask = ~((tpf.time > first_timestamp) & (tpf.time < last_timestamp))

                                    tpf = tpf[cadence_mask]

                                    additive_bkg = lk.DesignMatrix(tpf.flux[:, allfaint_mask]).pca(3)
                                    additive_bkg_and_constant = additive_bkg.append_constant()



                    # Now we correct all the bright pixels by the background, so we can find the remaining multiplicative trend

                    r = lk.RegressionCorrector(lk.LightCurve(tpf.time, tpf.time*0))

                    corrected_pixels = []
                    for idx in range(allbright_mask.sum()):
                        r.lc.flux = tpf.flux[:, allbright_mask][:, idx]
                        r.correct(additive_bkg_and_constant)
                        corrected_pixels.append(r.corrected_lc.flux)

                    #Getting the multiplicative effects now from the bright pixels that have been corrected for additive effects.
                    multiplicative_bkg = lk.DesignMatrix(np.asarray(corrected_pixels).T).pca(3)

                    #Create a higher order version of the additive effects:
                    additive_bkg_squared = deepcopy(additive_bkg)
                    additive_bkg_squared.df = additive_bkg_squared.df**2

                    #Now we make a fancy hybrid design matrix that has both orders of the additive effects and the multiplicative ones.

                    dm = lk.DesignMatrixCollection([additive_bkg_and_constant, additive_bkg_squared, multiplicative_bkg])

                    #Now get the raw light curve.
                    lc = tpf.to_lightcurve(aperture_mask=aper_mod)
                #  lc = lc[lc.flux_err > 0]        #This was suggested by an error message to prevent the "flux uncertainties" problem.

                    #Replace any errors that are zero or negative with the mean error:

                    mean_error = np.mean(lc.flux_err[np.isfinite(lc.flux_err)])
                    lc.flux_err = np.where(lc.flux_err == 0,mean_error,lc.flux_err)
                    lc.flux_err = np.where(lc.flux_err < 0,mean_error,lc.flux_err)

                    #And correct it:
                    clc = lk.RegressionCorrector(lc).correct(dm)

                    #NOW BEGIN PART WHERE I DO IT THE OLD FASHIONED WAY TO COMPARE:

                    raw_lc_OF = tpf.to_lightcurve(aperture_mask=aper_mod)

                    #Replace any errors that are zero or negative with the mean error:
                    raw_lc_OF.flux_err = np.where(raw_lc_OF.flux_err == 0,mean_error,raw_lc_OF.flux_err)
                    raw_lc_OF.flux_err = np.where(raw_lc_OF.flux_err < 0,mean_error,raw_lc_OF.flux_err)
                    raw_lc_OF.flux_err = np.where(np.isnan(raw_lc_OF.flux_err)==True,mean_error,raw_lc_OF.flux_err)

                #    raw_lc_OF = raw_lc_OF[raw_lc_OF.flux_err > 0]   #This was suggested by an error message to prevent the "flux uncertainties" problem.
                    regressors_OF = tpf.flux[:,~aper_mod]

                    dm_OF = lk.DesignMatrix(regressors_OF,name='regressors')
                    dm_pca5_OF = dm_OF.pca(3)
                    dm_pca5_OF = dm_pca5_OF.append_constant()
                    corrector_pca5_OF = lk.RegressionCorrector(raw_lc_OF)
                    corrected_lc_pca5_OF = corrector_pca5_OF.correct(dm_pca5_OF)

                #    model_pca5_OF = corrector_pca5_OF.model_lc
                    #model_pca5_OF -= np.percentile(model_pca5_OF.flux,5)
                #    corrected_lc_pca5_OF = raw_lc_OF - model_pca5_OF

                    #AND PLOT THE CORRECTED LIGHT CURVES, BOTH METHODS TOGETHER:
                    #Want to also add the mult/add component plots and the aperture plot!


                    fig2 = plt.figure(figsize=(12,8))
                    gs = gridspec.GridSpec(ncols=3, nrows=3,wspace=0.5,hspace=0.5,width_ratios=[1,1,2])
                    f_ax1 = fig2.add_subplot(gs[0, :])
                    f_ax1.set_title(target+': Corrected Light Curves')
                    f_ax2 = fig2.add_subplot(gs[1, :-1])
                    f_ax2.set_title('Additive Components')
                    f_ax3 = fig2.add_subplot(gs[2:,:-1])
                    f_ax3.set_title('Multiplicative Components')
                    f_ax4 = fig2.add_subplot(gs[1:,-1])
                    #f_ax4.set_title('Aperture')

                    clc.plot(ax=f_ax1,label='Hybrid Method')
                    corrected_lc_pca5_OF.plot(ax=f_ax1,label='PCA5 Simple')

                    f_ax2.plot(additive_bkg.values)
                    f_ax3.plot(multiplicative_bkg.values + np.arange(multiplicative_bkg.values.shape[1]) * 0.3)

                    tpf.plot(ax=f_ax4,aperture_mask=aper_mod,title='Aperture')

                    # print("\n")
                    print("\nMade lightcurve and aperture selection.\n")

                    ### Keeping track of what relative sector is being observed, keeps figures from experiencing FileExistsError
                    ## This section creates individual directories for each object in which the quaver procesed light curve data is stored
                    ##  then saves the corrected lightcurves along with additive and multiplicative components as well as the aperture selection

###############################################################################
##############################################################################
                    directory = str(target).replace(" ","")
                    target_safename = target.replace(" ","")
                    try:
                        os.makedirs('regression_program_output/'+target_safename)
                        print("Directory '% s' created\n" % directory)
                        plt.savefig('regression_program_output/'+target_safename+'/'+target_safename+'_regression_w_apsel_add_mult_comp_lcs_sector'+sec+'.pdf',format='pdf')
                        plt.show()
                    except FileExistsError:
                        print("Saving to folder '% s'\n" % directory)
                        plt.savefig('../regression_program_output/'+target_safename+'/'+target_safename+'_regression_w_apsel_add_mult_comp_lcs_sector'+sec+'.pdf',format='pdf')
                        plt.show()
##################################################################################
###############################################################################



                    regression_corrected_lc = np.column_stack((clc.time,clc.flux,clc.flux_err))
                    pca_corrected_lc = np.column_stack((corrected_lc_pca5_OF.time,corrected_lc_pca5_OF.flux,corrected_lc_pca5_OF.flux_err))

                    unstitched_lc_regression.append(regression_corrected_lc)
                    unstitched_lc_pca.append(pca_corrected_lc)

                    print("Sector, CCD, camera: ")
                    print(sector_number,ccd,cam)

#############################################
#############################################
                    print("\nMoving to next sector.\n")
#############################################
#############################################



        # If target coordinates are too close to edge on approach, this will skip that sector and read the next.
        # If target coordinates are too close to edge on exit, this will skip that sector and break on the next loop.
        ## WARNING: May also occur if connection to HEASARC could not be made. Check website and/or internet connection.

#############################################
#############################################
        except (http.client.IncompleteRead):

            print("Unable to download FFI cutout. Desired target coordinates may be too near the edge of the FFI.\n")
            print("Could be inability to connect to HEASARC. Check website availability and/or internet connection.\n")

            if i != num_obs_sectors-1:

              print("\nMoving to next sector.\n")

            continue

#############################################
#############################################

print("No more observed sectors in this cycle.")

print("Stitching light curves together.\n")

#Loop for stitching the light curves together


for j in range(0,len(unstitched_lc_regression)):
    if j==0:
        print("First observed sector")
    else:
        sector = str(j+1)
        print('Stitching '+sector+' sectors')

    lc_reg = unstitched_lc_regression[j]
    lc_pca = unstitched_lc_pca[j]

    t_reg = lc_reg[:,0]
    f_reg = lc_reg[:,1]
    err_reg = lc_reg[:,2]

    t_pca = lc_pca[:,0]
    f_pca = lc_pca[:,1]
    err_pca = lc_pca[:,2]

    if j == 0:

        full_lc_flux_reg = f_reg
        full_lc_flux_pca = f_pca

        full_lc_time_reg = t_reg
        full_lc_time_pca = t_pca

        full_lc_err_reg = err_reg
        full_lc_err_pca = err_pca

    else:

        first_flux_reg = f_reg[0]
        first_flux_pca = f_pca[0]

        last_flux_reg = full_lc_flux_reg[-1]
        last_flux_pca = full_lc_flux_reg[-1]

        scale_factor_reg = first_flux_reg - last_flux_reg
        scale_factor_pca = first_flux_pca - last_flux_pca

        if scale_factor_reg > 0:

            scaled_flux_reg = f_reg - abs(scale_factor_reg)

        if scale_factor_reg < 0:

            scaled_flux_reg = f_reg + abs(scale_factor_reg)

        if scale_factor_pca > 0:

            scaled_flux_pca = f_pca - abs(scale_factor_pca)

        if scale_factor_pca < 0:

            scaled_flux_pca = f_pca + abs(scale_factor_pca)


        full_lc_flux_reg = np.append(full_lc_flux_reg,scaled_flux_reg)
        full_lc_flux_pca = np.append(full_lc_flux_pca,scaled_flux_pca)

        full_lc_time_reg = np.append(full_lc_time_reg,t_reg)
        full_lc_time_pca = np.append(full_lc_time_pca,t_pca)

        full_lc_err_reg = np.append(full_lc_err_reg,err_reg)
        full_lc_err_pca = np.append(full_lc_err_pca,err_pca)



#Remove single-cadence jumps greater than 1% of the flux on either side from both finished light curves

for i in range(0,1-len(full_lc_time_reg)):

    if i !=0 and i != len(full_lc_flux_reg)-1 and full_lc_flux_reg[i] > (0.01 * full_lc_flux_reg[i-1]+full_lc_flux_pca[i-1]) and full_lc_flux_pca[i] > (0.01 * full_lc_flux_pca[i+1]+full_lc_flux_pca[i+1]):

        full_lc_flux_reg = np.delete(full_lc_flux_reg,i)
        full_lc_time_reg = np.delete(full_lc_time_reg,i)

for i in range(0,1-len(full_lc_time_pca)):

    if i !=0 and i != len(full_lc_flux_reg)-1 and full_lc_flux_pca[i] > (0.01 * full_lc_flux_pca[i-1]+full_lc_flux_pca[i-1]) and full_lc_flux_pca[i] > (0.01 * full_lc_flux_pca[i+1]+full_lc_flux_pca[i+1]):

        full_lc_flux_pca = np.delete(full_lc_flux_pca,i)
        full_lc_time_pca = np.delete(full_lc_time_pca,i)

for i in range(0,1-len(full_lc_time_reg)):

    if i !=0 and i != len(full_lc_flux_reg)-1 and full_lc_flux_reg[i] < (full_lc_flux_pca[i-1]-0.01 * full_lc_flux_reg[i-1]) and full_lc_flux_pca[i] < (full_lc_flux_pca[i+1]-0.01 * full_lc_flux_pca[i+1]):

        full_lc_flux_reg = np.delete(full_lc_flux_reg,i)
        full_lc_time_reg = np.delete(full_lc_time_reg,i)

for i in range(0,1-len(full_lc_time_pca)):

    if i !=0 and i != len(full_lc_flux_reg)-1 and full_lc_flux_pca[i] < (full_lc_flux_pca[i-1]-0.01 * full_lc_flux_pca[i-1]) and full_lc_flux_pca[i] > (full_lc_flux_pca[i+1]-0.01 * full_lc_flux_pca[i+1]):

        full_lc_flux_pca = np.delete(full_lc_flux_pca,i)
        full_lc_time_pca = np.delete(full_lc_time_pca,i)

#Compile and save the corrected light curves.

regression_lc = np.column_stack((full_lc_time_reg,full_lc_flux_reg,full_lc_err_reg))
pca5_lc = np.column_stack((full_lc_time_pca,full_lc_flux_pca,full_lc_err_pca))

np.savetxt('regression_program_output/'+target_safename+'/'+target_safename+'_full_hybrid_regressed_lc.dat',regression_lc)
np.savetxt('regression_program_output/'+target_safename+'/'+target_safename+'_full_pca5_lc.dat',pca5_lc)



#Plot the corrected light curves and save image.
plt.errorbar(full_lc_time_pca,full_lc_flux_pca,yerr = full_lc_err_pca,marker='o',markersize=1,color='orange',linestyle='none',label='PCA5 Method')
plt.errorbar(full_lc_time_reg,full_lc_flux_reg,yerr = full_lc_err_reg,marker='o',markersize=1,color='b',linestyle='none',label='Hybrid Method')

plt.legend()


for i in range(0,len(unstitched_lc_regression)):

    last_time = unstitched_lc_regression[i][:,0][-1]

    plt.axvline(x=last_time,color='k',linestyle='--')

plt.savefig('regression_program_output/'+target_safename+'/'+target_safename+'_full_corr_lcs.pdf',format='pdf')

plt.show()
print ("Done!")
