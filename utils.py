import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
import glob
import pickle
import corner
import scipy.optimize
import re
import os
import sys
import utils

#for loading TEPSPEC_Util scripts into trace pkl
sys.path.append('/Users/mango/tepspec/IMACS/')

from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from collections import Counter, namedtuple
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
from scipy import stats
from collections import OrderedDict

# pickle loader convenience function
def pkl_load(fn): 
    with open(fn, 'rb') as f: 
        data = pickle.load(f, encoding='latin') # Python 2 -> 3
    return data

# fits loader convenience function
def fits_data(fn): 
    with open(fn, 'rb') as f: 
        data = fits.open(f)[0].data
        return data
def fits_header(fn): 
    with open(fn, 'rb') as f: 
        header = fits.open(f)[0].header
        return header
    
# file save convenience function
def save_fig(dirname, fname):
    if dirname[-1] != '/': dirname += '/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    plt.savefig(dirname + fname, bbox_inches='tight')

# get index of array where val is nearest
def find_nearest_idx(array, value): 
    idx = (np.abs(array-value)).argmin()
    return idx

# return location (wav, depth) of spectra near loc
def find_crits(wav, val, loc, aper=24): 
    # first find wavlength of discretized spectra nearest loc
    wav_cen_idx = find_nearest_idx(wav, loc)
    
    # now make a 60 Angstrom window (aperture) around center 
    # and look for min.
    # this will hopefully be where the true absorption peak is.
    # note: a difference of 1 in index is 1.25 in Angstroms
    #peak_idx = np.where(val == val[wav_cen_idx-3:wav_cen_idx+4].min())[0][0]
    peak_idx = np.where(
            val == val[wav_cen_idx-aper:wav_cen_idx+aper+1].min())[0][0]
    # base_idx is the "floor" in a 100 A window, used for zoom plots
    #base_idx = np.where(val == val[wav_cen_idx-3:wav_cen_idx+4].max())[0][0]
    base_idx = np.where(
            val == val[wav_cen_idx-aper:wav_cen_idx+aper+1].max())[0][0]
    # the [0][0] just makes sure we take the first instance of the 
    # minimum that is
    # found in the window, in case there are double peaks
    return peak_idx, val[peak_idx], val[base_idx]

def plot_spec(ax, LC, time_idx, show_peaks=True, show_bins=True, 
        wbins_path='', lw=1, **kwargs):  
    # naD is a doublet but it looks like a single peak at our 
    # resolution so I am just going to take the average for this one
    """species = {
    '$NaI-D$'    : {'wav':5892.9, 'c':'lightblue' , 'dif':[], 'ls':'-'} ,
    r'$H\alpha$' : {'wav':6564.6, 'c':'g' , 'dif':[], 'ls':'-'} ,
    '$KI-a$'   : {'wav':7665.0, 'c':'y' , 'dif':[], 'ls':'-'} ,
    '$K1-$avg' : {'wav':7682.0, 'c':'b' , 'dif':[], 'ls':'-'} ,
    '$KI-b$'   : {'wav':7699.0, 'c':'y' , 'dif':[], 'ls':'--'} ,
    '$NaI-8200a$' : {'wav':8183.0, 'c':'r' , 'dif':[], 'ls':'-'} ,
    '$NaI-8200b$' : {'wav':8195.0, 'c':'r' , 'dif':[], 'ls':'--'}
    }"""
    
    species = OrderedDict()
    species['$NaI-D$'] = {'wav':5892.9, 'c':'lightblue', 
            'dif':[], 'ls':'-'} 
    species[r'$H\alpha$'] = {'wav':6564.6, 'c':'lightgreen', 
            'dif':[], 'ls':'-'} 
    species['$K1\_$avg'] = {'wav':7682.0, 'c':'lightyellow', 
            'dif':[], 'ls':'-'} 
    species['$NaI-8200\_avg$'] = {'wav':8189.0, 'c':'pink', 
            'dif':[], 'ls':'-'} 
    
    wav    = LC['spectra']['wavelengths']   # convert to nm
    transm = LC['spectra']['WASP43b'][time_idx]
    fwhm   = LC['fwhm'][b'WASP43b_8'][time_idx]
    t = Time(LC['t'], format='jd')
    
    for elem, elem_attr in species.items():
        ax.axvline(elem_attr['wav'], ls=elem_attr['ls'], 
                   c=elem_attr['c'], label=elem, lw=lw)
    # plot peak points
    if (show_peaks):
        for elem, elem_attr in species.items():

            pk_idx, pk_val, base_val = find_crits(wav, 
                    transm, elem_attr['wav'])
            ax.plot(wav[pk_idx], pk_val, 'o', color=elem_attr['c'])

    ax.plot(wav, transm, **kwargs)

    if (show_bins):
        wbins = ascii.read(wbins_path, 
                format='fixed_width_no_header', delimiter='\s')
        for wbin in wbins:
            row = wbin[0].split()
            l, r = float(row[0]), float(row[1])
            if (r-l) == 10 or (r-l) == 60 or (r-l) == 100:
                ax.axvspan(xmin=l, xmax=r, alpha=0.2)
    return ax

def plot_spec_diff(ax, LC):    
    # naD is a doublet but it looks like a single peak at our 
    # resolution so I am just going to take the average for this one
    species = {
    '$NaI-D$'    : {'wav':5892.9, 'c':'b' , 'dif':[], 'ls':'-'} ,
    r'$H\alpha$' : {'wav':6564.6, 'c':'g' , 'dif':[], 'ls':'-'} ,
    '$KI-a$'   : {'wav':7665.0, 'c':'y' , 'dif':[], 'ls':'-'} ,
    '$KI-b$'   : {'wav':7699.0, 'c':'y' , 'dif':[], 'ls':'--'} ,
    '$NaI-8200a$' : {'wav':8183.0, 'c':'r' , 'dif':[], 'ls':'-'} ,
    '$NaI-8200b$' : {'wav':8195.0, 'c':'r' , 'dif':[], 'ls':'--'}
    }
    
    # fill in dictionary 
    for i in range(len(LC['t'])): # iterate through time
        wav    = LC['spectra']['wavelengths'] # (1 x wavelengths)
        transm = LC['spectra']['WASP43b'][i,:] # (time x wavelength)
        t = Time(LC['t'], format='jd')
        
        for elem, elem_attr in species.items(): # iterate through species
            pk_idx, val, val_base = find_crits(
                    wav, transm, elem_attr['wav'], aper=4)
            dif = wav[pk_idx] - elem_attr['wav'] # measured - vacuum
            elem_attr['dif'].append(dif)
            
    # plot the differences
    #fig, ax = plt.subplots(2, 1, figsize=(8,5), sharex=False)
    
    for elem, elem_attr in species.items():
        ax.plot_date(t.plot_date, elem_attr['dif'], 
                     c=elem_attr['c'], linestyle=elem_attr['ls'], 
                     ms=0, label=elem)
        
    return ax

def plot_raw_LCs(ax, LC, plot_date=True, **kwargs):
    """Plots median normalized LCs for target and comparison stars.
    
    INPUTS
    ======
    ax:         matplotlib.axes object
                Empty axis to plot on
    LC:         pickle file
                LCs_<target>_<binsize>.pkl returned by tepspec
    target:     string 
                target star name
    plot_date:  boolean, default value is True
                whether to plot in units of time or time index

    RETURNS
    =======
    matplotlib.axes object: relative flux plot
        fills in the empty axis with a plot of the observed flux from the
        target star and each comparison star. All fluxes are relative to the
        median flux of the target star
    """
    t = Time(LC['t'], format='jd') # convert to astropy Time object

    # comp stars
    comp_name_sorted = sorted(LC['cNames'])
    for comp_name in comp_name_sorted:
        comp_idx = LC['cNames'].index(comp_name)
        lc_comp_i = LC['cLC'][:, comp_idx]
        lc_comp_i = lc_comp_i / LC['etimes']
        lc_comp_i = lc_comp_i / np.median(lc_comp_i)
        if plot_date:
            ax.plot_date(t.plot_date, lc_comp_i, '.', alpha=0.75, 
                         label=comp_name, **kwargs) 
        else:
            ax.plot(lc_comp_i, '.', alpha=0.75, label=comp_name)

    # target
    LC_targ = LC['oLC']
    LC_targ = LC_targ / LC['etimes']
    LC_targ = LC_targ / np.median(LC_targ)
    if plot_date:
        ax.plot_date(t.plot_date, LC_targ, '.', 
            lw=3, c='grey', alpha=0.75, label='WASP-43')
    else:
        ax.plot(LC_targ, '.', lw=3, c='grey', alpha=0.75, label='WASP-43')
    
    return ax

def plot_raw_comps(ax, LC, plot_date=True, **kwargs):
    # plot comparison star flux relative to median target flux
    t = Time(LC['t'], format='jd') # convert to astropy Time object 
    comp_name_sorted = sorted(LC['cNames'])
    for comp_name in comp_name_sorted:
        comp_idx = LC['cNames'].index(comp_name)
        lc_comp_i = LC['cLC'][:, comp_idx]
        lc_comp_i = lc_comp_i / (LC['etimes']*np.median(lc_comp_i))  
        #lc_comp_i /= LC['etimes']
        if plot_date:
            ax.plot_date(t.plot_date, lc_comp_i, '.', alpha=0.75, 
                         label=comp_name, **kwargs) 
        else:
            ax.plot(lc_comp_i, '.', alpha=0.75, label=comp_name)

    return ax

# comp stars to use, e.g. ['comp1', 'comp2']
def get_comp_divLC(LC, comp_names): 
    """ Calculates LC_adj = raw target LC / sum of selected comparison 
        star flux
    
    INPUTS
    ======
    LC:         pickle file
                LCs_<target>_<binsize>.pkl returned by tepspec
    comp_names:	string
		list of comparison stars to sum
    RETURNS
    =======
    LC_comp_div: namedtuple
        Contains the following fields,
        t:          astropy Time object
        comp_div:   median normalized LC_adj
        comp:       total raw comparison star flux over time
			
    """
    t = Time(LC['t'], format='jd')
    t = t.datetime # convert to UTC day:hour:min:sec
        
    comp_flux = 0 # running total of good comp star flux
    for comp in comp_names: 
        # use .index to handle cNames not in numerical order
        comp_idx = LC['cNames'].index(comp)
        comp_flux += LC['cLC'][:, comp_idx]
        
    LC_adj = LC['oLC'] / comp_flux
    
    LC_comp_div = namedtuple('LC_comp_div', 't comp_div comp')
        
    return LC_comp_div(t, LC_adj / np.median(LC_adj), comp_flux)

# computes LC / (sum comp flux) on a per bin basis
def get_comp_divLC_w(LC, comps_to_use):
    """ Calculates LC_comp_div = (raw target LC / sum of selected comparison 
        star flux) per wavelength bin
    
    INPUTS
    ======
    LC:         pickle file
                LCs_<target>_<binsize>.pkl returned by tepspec
    comp_names:	string
		list of comparison stars to sum
    RETURNS
    =======
    LC_comp_div_w: nested list of LC_comp_div where each element of the top
    list is for a wavelength bin, ordered from smallest to largest
    """

    # for residual calculation
    wlc = get_comp_divLC(LC, comps_to_use).comp_div    

    # assumes wav_bins are in ascending order
    N = len(LC['wbins'])
    M = len(LC['t'])
    LCs = np.empty([N, M])
    resids = np.empty([N,M])
    LC_list = [] # each element will hold LC_o / sum(LC_comp) per bin
    residuals = []
    for w in range(len(LC['wbins'])):
        comp_flux = 0
        for comp in comps_to_use: 
            # use .index to handle cNames not in numerical order
            comp_idx = LC['cNames'].index(comp)
            comp_flux += LC['cLCw'][:, comp_idx, w]        
        # don't divide if total comp flux is zero
        if (np.sum(comp_flux) == 0): comp_flux = 1
        lc_w = LC['oLCw'][:, w] / comp_flux
        lc_w /= np.median(lc_w)
        #LC_list.append(lc_w) # save LC per bin
        LCs[w] = lc_w
        resid = lc_w - wlc + 1 # plus one for lining up with binned LC plot
        #residuals.append(resid)
        resids[w] = resid

    LC_comp_div_w = namedtuple('LC_comp_div_w', 't wbins flux resid_wlc')
        
    return LC_comp_div_w(LC['t'], LC['wbins'], LCs, resids)

def plot_LC_comp_div(ax, t, flux, comps_to_use, target, 
        date, binsize, idx_mask=[], utc=True, **kwargs):
    """ plots WLC returned by utils.get_comp_divLC
    """
    comps_used = 'Using: ' + str(comps_to_use)
    ax.annotate(comps_used, xy=(0.5, 0.95), xycoords='axes fraction', 
    ha='center', fontsize=7.5, va='top')
    t = np.delete(t, idx_mask) # also accepts idx_mask=[] to not remove pts 
    flux = np.delete(flux, idx_mask)
    if utc:
        ax.plot(t, flux, **kwargs)
        ax.set_xlabel('Time (UTC)')
    else:
        ax.plot(flux, **kwargs)
        ax.set_xlabel('Time index')
    #ax.plot(flux, 'ro')
    #ax_idx = ax.twiny()
    #ax.plot(flux, 'ro')
    #ax_idx.set_xticks(np.arange(0, len(flux), idx_skip))
    #ax_idx.grid(False)
    #ax.set_title('{} {} {} Divided WLC'.format(target, date, binsize))
    #ax.set_ylabel('Relative Flux')
    #ax.legend()
   
    return ax

def plot_binned(ax, time, flux, bins, offs, cmap, utc=True,
                ignore_last_bin=False, annotate=True, hue=1, **kwargs): 
    # e.g. dataset: 'ut150309', model: 'poly', offs: 0.01, const: 7 minutes
    N = len(bins) # number of wavelength bins
    if ignore_last_bin: N = N-1
    colors = np.array(sns.color_palette(cmap, N))
    # hold vacuum wavelengths of certain species to be highlighted
    species = {r'$NaI-D$':5892.9, r'$H\alpha$':6564.6, 
               '$K\mathrm{I}_\mathrm{avg}$':7682.0, '$Na\mathrm{I}\ 8200_\mathrm{avg}$':8189.0}         
    offset = 0
    for i in range(N):
        wav_bin = [round(bins[i][j], 3) for j in range(2)]
        mec = 0.7 * colors[i] # controls saturation of edge colors
    
        if (utc):
            t_date = Time(time, format='jd')
            ax.plot_date(t_date.plot_date, flux[i] + offset, c=hue*colors[i], 
                    mec=mec, label = wav_bin, **kwargs)
        else:
            ax.plot(flux[i] + offset, c=hue*colors[i], 
            mec=mec, label = wav_bin, **kwargs)

        if (annotate):
            ann = ax.annotate(wav_bin, xy=(time[0], 1.004*(1 + offset)),
            fontsize=8,
            path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")])
            
            for spec, spec_wav in species.items():
                if (spec_wav < wav_bin[1]) and (spec_wav > wav_bin[0]):
                    #ann.set_color('b')
                    ann.set_text('{} - {}'.format(ann.get_text(), spec))
                    ann.set_weight('bold')
            
        offset += offs
         
    return ax

# extracts stats from pkl file
def get_detr_stats(pkl): 
    # time
    idx_fit = pkl['idx_fit'][0] # same for all bin(s)
    t0 = pkl['t0']
    #pkl_t0_iso = Time(t0, format='jd').iso # iso format
    t = pkl['t']
    t_idx = np.arange(t.size)
    t_rel = (t - t0) * 24. # convert to hours
    #t_rel = t_rel[idx_fit]

    # get array dimensions
    N = len(pkl['detLC']) # number of rows
    #M = len(idx_fit) # number of columns
    M = len(t)
    
    # wavelength bins
    wbins = np.empty([N, 2])
    for k, val in pkl['wbins'].items():
        wbins[k] = val
        
    # detrended model
    model = np.empty([N, M])
    for k, val in pkl['model'].items():
        model[k] = val #val[idx_fit]

    # detrended data point
    detLC = np.empty([N, M])
    for k, val in pkl['detLC'].items():
        detLC[k] = val
    
    # calculate residuals
    resid = np.empty([N, M])
    resid_elvis = np.empty([N, M])
    #for k, val in detLC.items():
    #    resid[k] = detLC - model
    #    # should be the same as resid, just a check
    
    resid = detLC - model #model
    resid_elvis[k] = pkl['residuals'][k]
        
    pkl_stats = namedtuple('pkl_stats', 't_idx t_rel idx_fit wbins detLC model \
                           resid resid_elvis')
    
    return pkl_stats(t_idx, t_rel, idx_fit, wbins, detLC, model, resid, resid_elvis)

# return fitted RpRs_unc (Gaussian sigma) for given PD file
# which is for given wavbin
# NOTE: replace this with UCIs-meds, meds-LCIs from posteriors.pkl later.
# This keeps all the stats info in one place (less files to download)
# and now I don't need to assume symmetric error bars = D
def get_sigma(fpaths): 
    sigma_list = []
    for fpath in sorted(glob.glob(fpaths)): # loop over wav bins
        pd = utils.pkl_load(fpath)
        RpRs = pd['RpRs']
        mu, sigma = stats.norm.fit(RpRs) # get stats for error bars
        sigma_list.append(sigma)    
        
    return np.array(sigma_list)

def get_rprs_unc(bin_list, path):
    rprs_med_list = []  # -->
    rprs_mean_list = [] # saving both for diagnostic purposes
    rprs_unc_list = []
    for b in bin_list: # loop over bin sizes
        fn = '{}/{}/PD*'.format(path, b)
        rprs_unc = get_sigma(fn) # get rprs_unc for wth bin
        rprs_unc_list.append(rprs_unc)
    
    return rprs_unc_list

# Gets rprs, wavbin centers, and wavbin widths (wavbin error).
# The posteriors are ordered by average wavelength 
# $(\lambda_{min}+\lambda_{max})/2$ when saved to the pickle.
def get_rprs(bin_list, path):
    wav_list = []
    wav_err_list = []
    rprs_list = []

    #fpaths = path + '*nm/posteriors.pkl'
    #for fpath in sorted(glob.glob(fpaths)):
    for b in bin_list:
        fpath = '{}/{}/posteriors.pkl'.format(path, b)
        post = utils.pkl_load(fpath)
        #print post.keys()
        wav_avg = (post['wl_maxs'] + post['wl_mins']) / 2 # center of bins
        binsize = post['wl_maxs'] - post['wl_mins'] # for horizontal errorbars
        RpRs_med = post['RpRs']['meds'] # median

        wav_list.append(wav_avg)
        wav_err_list.append(binsize)
        rprs_list.append(RpRs_med)
    
    return wav_list, wav_err_list, rprs_list

def trans_spec_plot(ax, bin_list, color_list, path, fmt):
    # plot reference absorption lines
    wav_name = np.array(['NaI-D','H-alpha','KI_avg','NaI-8200_avg'])
    wav_cen = np.array([5892.9, 6564.6, 7682.0, 8189.0])
    base_cen = np.array([5580.0, 5580.0, 8730.0, 8730.0])
    [ax.axvline(i, c='grey', alpha=0.5, ls='--') for i in wav_cen]

    # get plot values
    wav_list, wav_err_list, rprs_list = get_rprs(bin_list, path)
    rprs_unc_list = get_rprs_unc(bin_list, path)
    #print get_rprs_unc(bin_list, path)
    ars = [wav_list, wav_err_list, rprs_list, 
            rprs_unc_list, bin_list, color_list]
    for wav, wav_err, rprs, rprs_err, lb, c in zip(*ars):
        ax.errorbar(wav, rprs, xerr=wav_err/2, yerr=rprs_err, fmt=fmt, 
                   label=lb, alpha=0.6, color=c)
    #ax.set_ylim(0.15, 0.164)

    ax.set_xlabel('Wavelength $(\AA)$')
    ax.set_ylabel(r'$R_p/R_s$')
    ax.legend(loc='best')
    
    return ax

def plot_corner(params, mcmc, **kwargs):  
    # creates another dictionary that just has the entries named in params 
    sub_param_dict = {k:v for k, v in mcmc.items() if k in params}
    # merge all chains into 1 for each param
    for k, v in sub_param_dict.items():
        sub_param_dict[k] = np.concatenate(list(v.values()))
    # order sub_param_dict based on params order and save in param_dict
    param_dict = {p:sub_param_dict[p] for p in params}
    # create (nsamples x nparams) array for corner plot
    samples = np.column_stack((list(param_dict.values())))
    # convert labels to latex math mode
    if 'RpRs' in params: 
        params[params.index('RpRs')] = r'$R_\mathrm{p} / R_\mathrm{s}$'
    if 'a0' in params: 
        params[params.index('a0')] = r'$a_0$'
    if 'a1' in params: 
        params[params.index('a1')] = r'$a_1$'
    if 'a2' in params: 
        params[params.index('a2')] = r'$a_2$'
    if 'sigma_w' in params: 
        params[params.index('sigma_w')] = r'$\sigma_\mathrm{w}$'
    if 'sigma_r' in params: 
        params[params.index('sigma_r')] = r'$\sigma_\mathrm{r}$'
    p = corner.corner(samples, labels=params, show_titles=True, 
            use_math_text=True, title_fmt='.3g',
            **kwargs)
    return p

def plot_bintest(ax, bin_info_x, name_x, c='g', n=2):
    # plot error bar
    sorted_bin_info_x = sorted(bin_info_x[name_x]['RpRs'].items())
    binsize, rprs = zip(*sorted_bin_info_x)
    med = np.array([r['med'] for r in rprs])
    upper = np.array([r['UCI'] for r in rprs]) - med
    lower = med - np.array([r['LCI'] for r in rprs])
    ax.errorbar(binsize, med, yerr=[lower, upper], 
                fmt='o', c=c, alpha=0.5)
    
    # plot fitted region
    p, C_p = np.polyfit(binsize, med, n, cov=True, 
                        w=1./np.mean([lower, upper], axis=0))
    t = np.linspace(np.min(binsize), np.max(binsize), 500)
    TT = np.vstack([t**(n-i) for i in range(n+1)]).T
    yi = np.dot(TT, p)
    C_yi = np.dot(TT, np.dot(C_p, TT.T)) # C_y = TT*C_z*TT.T
    sig_yi = np.sqrt(np.diag(C_yi))
    #sig_yi = np.abs(np.dot(TT, p+np.sqrt(np.diag(C_p))) - yi) # wrong
    ax.fill_between(t, yi + sig_yi, yi - sig_yi, alpha=0.25, color=c)
    ax.plot(t, yi, c=c)
    return ax
