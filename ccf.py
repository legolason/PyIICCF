#A code to measure the time lag in reverberation mapping
#Last modified on 6/20/2022
#Auther: Hengxiao Guo (UCI), Aaron Barth (UCI)
#Email: hengxiaoguo AT gmail DOT com, barth AT uci DOT edu
#version 1.3

#Main function: measure the time lag (including lag uncertainty and significance) between
#AGN continuum and emission line in reverberation mapping project. All the CARMA procedures
#are from Kelly Brandon's carma_pack.

#We thanks for the helpful discussion with Mouyuan Sun, Jennifer I. Li.

import numpy as np
import carmcmc as cm
import scipy.stats as sst
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from contextlib import contextmanager
import glob, os, sys, timeit, warnings
from sklearn.neighbors import KernelDensity
from multiprocessing import Lock, Pool, Manager
warnings.filterwarnings("ignore")


# setup the styles of output figures
plt.rcParams['figure.figsize']  = 8, 6
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.linewidth'] = 2



@contextmanager
def quiet():
    """Disable print in carma code."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class CCF(): 
    def __init__(self):
        pass
    
    def ICCF(self, t1, y1, e1, t2, y2, e2, tau_min = -100, tau_max = 100, step = 0.2, detrend = 0, interp = 'linear', mcmc_nsamples = 20000, auto_pq = False, p1 = 1, q1 = 0, p2 = 1, q2 = 0, carma_model = 'random', sig_cut = 0.8, imode = 0, MC_ntrials = 1000, FR_RSS = 0, sigmode = 0.2, weight = False, nsmooth_wgts = 1, sim_ntrials = 1000, sim_mode = 0, sim_var_range = 0.2, Nmodel = 2, MP = True, name = 'results', plotLC = True, shift = 'centroid', plotCCF = True, save = True, lite = True, path = './'):
        '''
        Main function will call other functions. This function is able to detrend the light curve, measure the time lag, 
        evaluate the lag uncertianties with Flux Randomization (FR) and Random Subset Sampling (RSS) and the significance with simulated CARMA Light Curves (LC).
        
        Input:
        t1, y1, e1: array_like
            time, flux and uncertainty for continuum light curve.
        t2, y2, e2: array_like
            time, flux and uncertainty for emission-line light curve.
        tau_min, tau_max: float
            maximum and minimun range to calculate the time delay.
        step: float    
            tau shifting step.
        detrend: interger 
            detrend the data with a polynomial function np.polyfit(t, y, detrend). Not detrend if 0.
        interp: 'linear' or 'carma'
            linear interpolation for all procedures (measuring lag, lag uncertainties and significance) if 'linear', CARMA-model interpolation for all procedures if   'carma', i.e., interpolate light curve with Damped Random Walk model (CARMA(p, q) with p = 1, q = 0). See details in Kelly et al. 2009, ApJ, 698, 895 and Kelly et al. 2014, ApJ, 788, 33.
        mcmc_nsamples: interger 
            N sampeles in CARMA MCMC process. 
        auto_pq: bool
            choose the best p and q automatically by CARMA model for continuum and emission line LCs if True. We search the all the grids with p <= 6 and q <= 5. This step will take much time.
        p1, q1, p2, q2: interger
            p1, q1 for continuum LC and p2, q2 for emission line LC. p > q required. DRW model if p = 1 and q = 0.
        carma_model: 'mean' or 'random'
            interpolate data with the sample mean carma model if 'mean', else we randomly draw a realization, which is more close to the reality.
        sig_cut: float
            0 < sig_cut <1. The required significance of the correlation coefficient r_max to calculate the CCF centroid, i.e., r_max * sig_cut
        FR_RSS: 0, 1 or 2
            use both FR and RSS to calculate the uncertainties if 0. Only use FR (RSS) if 1 (2).
        imode: 0, 1 or 2
            0: cross-correlation mode (two ways); 1: interpolate continuum LC; 2: interpolate emission-line LC.
        sigmode: float
            the required CCF r_max to calculate the lag uncertainties, 0<sigmode<1. r_max < sigmode will be considered as failed CCFs.
        MC_ntrials: interger
            n (>= 0) trials of Monte Carlo process to evaluate the lag uncertainties.
        sim_ntrials: interger
            n (>= 0) trials of light curve simulation. Only applied when SIM is True.
        weight: Bool
            apply the weights to lag posterier if True. Weights are decided by ACF and overlapping points. Our method is similar to Grier+19. See the details in Grier et al. 2019, ApJ, 887, 38
        nsmooth_wgts: > 0
            a scale factor multiply on the Gaussain bandwidth according to Scott's Rule n^(-0.2) [sigma], n is number of data points, for the lag posterier to decide the region (or local minimum around the primary peak) to calculate the lag and its uncertainties.
        sim_ntrials: interger
            n (even number, >2) trials of simulation to evaluate the lag significance. 0.5n trials are calculating the CCF between real continuum LC and simulated emission line LC, and the rest 0.5 is opposite.
        Nmodel: interger
            n (even number, >=2) models to use to produce the simulated LCs. If n =2, one model to produce the simulated continuum LCs and another one is for the simulated emission-line LCs.
        sim_var_range: float
            the maximum fraction of the variability of the mock LCs can deviate from the actual variability in the real LCs. E.g., the standard deviation of LC 1 is 0.5, then the simulated variability should be between 0.5*(1-x) and 0.5*(1+x).
        sim_mode: 0 or 1
            if 0, significance test will calculate CCF with real y1 and simualted y2, then real y2 and simulated y1. If 1, CCF will be based on both simulated y1 and y2. 
        MP: Bool
            Multiprocessing with all CPUs if true.
        name: string
            name of saved figures and fits file.
        plotLC: bool
            plot orignial LC (and CARMA model) if True.
        shift: 'peak' or 'centroid'
            shift the emission line LC with peak or centroid lag.
        plotCCF: bool
            pllt CCF results if True.
        save: Bool
            save figures and CCF results into fits if ture. MC_ntrial>1, sim_ntrials >2 are needed.
        lite: Bool
            save a lite version of the fits file.
        path: string
            path of the saved fits and figures.
    
            
        Retrun:
        t1, y1, e1, t2, y2, e2: 
            orignial or detrended (if detrend = True) light curves.
        lag, r:
            time lag array and averaged CCF pearson r array with two ways.
        lag_cen, lag_peak:
            actual CCF results of lag centroid and peak values.
        lag_1, r_1, lag_2, r_2:
            one-way time lag and CCF results. 1: interpolate y1. 2: interpolate y2.
        npt, npt_1, npt_2:
            overlapping points in CCF, npt is averaged and npt_1 (npt_2) is one-way result.
        p1, q1, p2, q2:
            the assumed CARMA model. The best p and q if auto_pq is True, 1 for continuum and 2 for emission line.
        var_sn1, var_sn2:
            sginal-to-noise ratio of the variability for continuum and emission-line light curves sqrt(chi^2-DOF), chi^2 = SUM [(Xobs-Xmedian)^2/sig^2] and DOF = N_lightcuve-1.      
        sample1, sample2:
            CARMA MCMC sample for y1 (sample1) and y2 (sample2). see details in Kelly Brandon's carma_pack.
        failed_CCF_MC:
            number of CCF result hits the tau boundaries (i.e., tau_min or tau_max). 
        peak_pack_MC, cen_pack_MC:
            lag and its error (1 sigma) from Monte Carlo method, i.e., [lower_error, best, upper_error].
        lag_peak_MC, lag_cen_MC:
            lag peak and centroid in flux randomization with Monte Carlo method. Failed CCF results have been
            removed.
        p_positive:
            fraction of positive simulated CCF peak larger than real-data CCF peak at tau >0, i.e., N1(tau>0&r>r_max_data)/N0(tau>0)
        p_all:
            fraction of simulated CCF peak larger than real-data CCF peak at any tau, i.e., N2(r>r_max_data)/N0
        p_peak:
            fraction of simulated CCF peak larger than real-data CCF peak at peak bin.
        lag_sim, lag_sim_cen, lag_sim_peak, r_sim, r_sim_cen, r_sim_peak
            lag_sim and r_sim saved all the simlated CCF curves. lag_sim_cen, lag_sim_peak, r_sim_cen and r_sim_peak are the corresponding
            peak/centroid lag and r.
        peak_left, peak_right, cen_left, cen_right:
            left and right boundaries of the effctive region used to calculated lag peak/centroid when weights applied.
        peak_kde, cen_ked:
            smoothed peak/centroid lag MC posterior profile.
        '''
    
        self.t1 = np.asarray(t1, dtype = np.float64)
        self.y1 = np.asarray(y1, dtype = np.float64)
        self.e1 = np.asarray(e1, dtype = np.float64)
        self.t2 = np.asarray(t2, dtype = np.float64)
        self.y2 = np.asarray(y2, dtype = np.float64)
        self.e2 = np.asarray(e2, dtype = np.float64)
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.step = step
        self.detrend = detrend
        self.interp = interp
        self.p1, self.q1, self.p2, self.q2 = p1, q1, p2, q2
        self.carma_model = carma_model
        self.imode = imode
        self.mcmc_nsamples = mcmc_nsamples
        self.sig_cut = sig_cut
        self.MC_ntrials = MC_ntrials
        self.sim_ntrials = sim_ntrials
        self.weight = weight
        self.sim_var_range = sim_var_range
        self.MP = MP
        self.plotLC = plotLC
        self.plotCCF = plotCCF
        self.name = name
        self.save = save
        self.path = path
        self.FR_RSS = FR_RSS
        self.sigmode = sigmode
        self.lite = lite
        self.Nmodel = Nmodel
        self.nsmooth_wgts = nsmooth_wgts
        self.shift = shift
        self.sim_mode = sim_mode
        
        
        # check parameters
        if self.Nmodel <2 and self.Nmodel > self.sim_ntrials:
            raise Exception("Number of models should be between 2 and sim_ntrials!") 
            
        if self.interp != 'linear' and self.interp != 'carma':
            raise Exception("interp must be 'linear' or 'carma'!") 
        
        # check the light curve  
        if t1.shape[0] <10 or t2.shape[0] <10:
            raise Exception("The light curve should contain at least 10 data points!") 
            
        # avoid nan, inf in data
        sm =  np.sum(np.isnan(self.t1)) + np.sum(np.isnan(self.t2)) + np.sum(np.isnan(self.y1)) + np.sum(np.isnan(self.y2))+ np.sum(np.isnan(self.e1)) + np.sum(np.isnan(self.e2))+ np.sum(np.isinf(self.t1)) + np.sum(np.isinf(self.t2))+ np.sum(np.isinf(self.y1)) + np.sum(np.isinf(self.y2)) + np.sum(np.isinf(self.e1)) + np.sum(np.isinf(self.e2))
        if sm >0:
            raise Exception("The light curve should not contain nan or inf data points!") 
            
        # avoid zero in errors        
        if 0. in self.e1:
            self.e1 = self.e1 + self.y1*1e-6 
        if 0. in self.e2:
            self.e2 = self.e2 + self.y2*1e-6 
        
        # sort the data according to date if they are not inceasing
        if np.sum(np.diff(self.t1)<0.0)>0 or np.sum(np.diff(self.t2)<0.0)>0:
            ind1 = np.argsort(self.t1)
            ind2 = np.argsort(self.t2)
            self.t1, self.y1, self.e1 = self.t1[ind1], self.y1[ind1], self.e1[ind1]
            self.t2, self.y2, self.e2 = self.t2[ind2], self.y2[ind2], self.e2[ind2]
            
        # calculate the variability SNR
        self.var_sn1 = np.sqrt(np.sum(((self.y1-np.median(self.y1))**2/self.e1**2)) - (len(self.y1)-1))
        self.var_sn2 = np.sqrt(np.sum(((self.y2-np.median(self.y2))**2/self.e2**2)) - (len(self.y2)-1))
        if np.isnan(self.var_sn1):
            self.var_sn1 = 0.0
        if np.isnan(self.var_sn2):
            self.var_sn2 = 0.0
        
            
        # choose the best p&q in carma model automatically for the LCs
        if auto_pq == True:
            # continuum
            with quiet():
                model = cm.CarmaModel(self.t1, self.y1, self.e1)
                sample = model.run_mcmc(self.mcmc_nsamples)
                best_MLE, pqlist, AICc = model.choose_order(6, njobs = -1, ntrials = 50)
            ind_finite = np.where(~np.isnan(AICc) & np.isfinite(AICc))
            self.p1, self.q1 =  np.array(pqlist)[ind_finite][np.argmin(np.array(AICc)[ind_finite])]
            
            # for emssion-line
            with quiet():
                model = cm.CarmaModel(self.t2, self.y2, self.e2)
                sample = model.run_mcmc(self.mcmc_nsamples)
                best_MLE, pqlist, AICc = model.choose_order(5, njobs = -1, ntrials = 50)
            ind_finite = np.where(~np.isnan(AICc) & np.isfinite(AICc))
            self.p2, self.q2 =  np.array(pqlist)[ind_finite][np.argmin(np.array(AICc)[ind_finite])]
            


        # calculate ICCF with linear or CARMA model   
        global N
        N = 0 # for actual data
        self.lag, self.r, self.npt = self.ICCF_CARMA(self.t1, self.y1, self.e1, self.t2, self.y2, self.e2, tau_min = self.tau_min, tau_max = self.tau_max, 
                                            step = self.step, interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, q1 = self.q1, 
                                            p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        N = 1 # for MC or simulations
        # do MC to estimate the lag error
        if self.MC_ntrials>0:
            # apply the weights to lag posterier
            if self.weight == True:    
                self.lag_ACF, self.r_ACF, self.overlap_npt, self.weighting = self.Weight()
            # estimate the lag error
            self.peak_pack_MC, self.cen_pack_MC, self.lag_peak_MC, self.lag_cen_MC = self.MonteCarlo(self.MC_ntrials)
           
        # do the LC simulation to calculate the lag significance
        if self.sim_ntrials>1:
            self.lag_sim, self.r_sim = self.CalSig(self.sim_ntrials)

        # plot figures
        if self.plotLC == True:
            self.PlotLC()
            
        if self.plotCCF == True:
            self.PlotCCF()
        # save results
        if self.save == True:
            self.SaveResults()
    
    def Detrend(self, t1, y1, t2, y2):
        # detrend with nth polynomial  
        T = np.linspace(t1.min(), t1.max(), 1000)
        cont = np.poly1d(np.polyfit(t1, y1, self.detrend))
        cont_new = np.interp(t1, T, cont(T))
        y1 = y1-cont_new+cont_new.mean()
        T = np.linspace(t2.min(), t2.max(), 1000)
        line = np.poly1d(np.polyfit(t2, y2, self.detrend))
        line_new = np.interp(t2, T, line(T))
        y2 = y2-line_new+line_new.mean()
        return t1, y1, t2, y2
        
    def ICCF_CARMA(self, t1, y1, e1, t2, y2, e2, tau_min = -100, tau_max = 100, step = 0.2, interp = 'linear', 
                   mcmc_nsamples = 20000, p1 = 1, q1 = 0, p2 = 1, q2 = 0, carma_model = 'random', imode = 0):
        """
        Calculate the time lag between continuum and emission-line LCs. 
        This function returns the lag, CCF r and number of data points used in CCF
        """
        np.random.seed()
        # If detrend > 0 do detrend for both LCs
        if self.detrend > 0 :
            t1, y1, t2, y2 = self.Detrend(t1, y1, t2, y2)
        
        
        # calculate ICCF
        # one bug: in carma p and q need to be reset
        lag_1, r_1, npt_1, lag_2, r_2, npt_2, = ([] for i in range(6))
        
        #prepare carma mcmc samplers
        if interp == 'carma':
            # setup the samples for y1, y2 with carma model
            with quiet():
                model2 = cm.CarmaModel(t2, y2, e2, p = self.p2, q = self.q2) #use self.p self.q since the carma bug
                sample2 = model2.run_mcmc(mcmc_nsamples)
            
                model1 = cm.CarmaModel(t1, y1, e1, p = self.p1, q = self.q1)
                sample1 = model1.run_mcmc(mcmc_nsamples)
                if N == 0:
                    # for actual data save samples
                    self.sample2 = sample2
                    self.sample1 = sample1
            
        # interpolate y2
        if imode!= 1:  #imode = 0, 2
            if interp == 'carma':
                time = np.linspace(t2.min(), t2.max(), int((t2.max()-t2.min())/step))
                if carma_model == 'mean':
                    model_mean, model_var = sample2.predict(time, bestfit = 'map')
                    t2_tmp, y2_tmp = time, model_mean
                if carma_model == 'random':
                    ysim = sample2.simulate(time, bestfit = 'map')
                    t2_tmp, y2_tmp = time, ysim  
            else:
                # for linear interpolation
                t2_tmp, y2_tmp = t2, y2
            
            # calculate pearson r
            tau = tau_min
            while tau < tau_max:
                lag_2.append(tau)
                t2_new = t2_tmp-tau
                ind_t1 = np.where( (t1 >=  t2_new.min()) & (t1 <=  t2_new.max()), True, False)
                npt_2.append(np.sum(ind_t1)) # "real" data points used in iccf
                if np.sum(ind_t1) > 0:
                    y2_new = np.interp(t1[ind_t1], t2_new, y2_tmp)
                    r, p = sst.pearsonr(y1[ind_t1], y2_new)
                    r_2.append(r)
                    tau = tau+step
                else:
                    raise Exception("No overlaping data in two light curves! Reset the search range!")
            
            lag_2 = np.array(lag_2)   
            r_2 = np.array(r_2)
            npt_2 = np.array(npt_2)
            
            if N  ==  0:
                self.lag_2 = lag_2
                self.r_2 = r_2
                self.npt_2 = npt_2
        
        # interpolate y1
        if imode != 2: # imode = 0, 1
            if interp == 'carma':
                time = np.linspace(t1.min(), t1.max(), int((t1.max()-t1.min())/step))
                if carma_model == 'mean':
                    model_mean, model_var = sample1.predict(time, bestfit = 'map')
                    t1_tmp, y1_tmp = time, model_mean
                if carma_model == 'random':
                    ysim = sample1.simulate(time, bestfit = 'map')
                    t1_tmp, y1_tmp = time, ysim
            else:
                t1_tmp, y1_tmp = t1, y1
            
            tau = tau_min
            while tau <tau_max:
                lag_1.append(tau)
                t1_new = t1_tmp+tau
                ind_t2 = np.where( (t2 >=  t1_new.min()) & (t2 <=  t1_new.max()), True, False)
                npt_1.append(np.sum(ind_t2))
                if np.sum(ind_t2) >0:
                    y1_new = np.interp(t2[ind_t2], t1_new, y1_tmp)
                    r, p = sst.pearsonr(y2[ind_t2], y1_new)
                    r_1.append(r)
                    tau = tau+step
                else:
                    raise Exception("No overlaping data in two light curves! Reset the search range!")

            lag_1 = np.array(lag_1)
            r_1 = np.array(r_1)
            npt_1 = np.array(npt_1)
            if N  ==  0:
                self.lag_1 = lag_1
                self.r_1 = r_1
                self.npt_1 = npt_1 
        
        # for different cases 
        if imode == 0:
            lag = (lag_1+lag_2)*0.5
            r = (r_1+r_2)*0.5
            npt = (npt_1+npt_2)*0.5
        elif imode == 1:
            lag = lag_1
            r = r_1
            npt = npt_1
        elif imode == 2:
            lag = lag_2
            r = r_2
            npt = npt_2
        else:
            raise Exception("Please select imode = 0, 1 or 2!") 

        
        # get the peak and centroid of data-based CCF
        if N  ==  0:
            self.lag_peak, self.lag_cen, self.r_peak, self.r_cen = self.Lag_center(lag, r)
        return lag, r, npt
    
    
    def Lag_center(self, lag, r):
        """
        Calculate the peak and centroid of lag and r according to sig_cut.
        """
        #peak
        np.random.seed()
        r_peak = max(r)
        if N == 0:
            self.rmax = r_peak
        r_peak_ind = np.argmax(r)
        lag_peak = lag[r_peak_ind]
        
        # centroid, we use the region contains the peak
        try:
            spline = interpolate.UnivariateSpline(lag, r-self.sig_cut*r.max(), s = 0.)
            left = spline.roots()[np.where(lag_peak>spline.roots(), True, False)].max()
            right = spline.roots()[np.where(lag_peak<spline.roots(), True, False)].min()
            if N == 0:
                self.left, self.right = left, right
            ind = np.where( (lag <= right.min()) & (lag >= left.max()), True, False)
            lag_cen = np.sum(r[ind]*lag[ind])/np.sum(r[ind])
            r_cen = r[np.where(lag  ==  lag[min(range(len(lag)), key = lambda i: abs(lag[i]-lag_cen))], True, False)][0]
        except:
            try:
                spline = interpolate.UnivariateSpline(lag, r-0.95*r.max(), s = 0.)
                left = spline.roots()[np.where(lag_peak>spline.roots(), True, False)].max()
                right = spline.roots()[np.where(lag_peak<spline.roots(), True, False)].min()
                
                if N == 0:
                    self.left, self.right = left, right
                ind = np.where( (lag <= right.min()) & (lag >= left.max()), True, False)
                lag_cen = np.sum(r[ind]*lag[ind])/np.sum(r[ind])
                r_cen = r[np.where(lag  ==  lag[min(range(len(lag)), key = lambda i: abs(lag[i]-lag_cen))], True, False)][0]
            except:
                # failed CCF
                lag_cen, r_cen = -9999., -9999.
                if N == 0:
                    self.right, self.left = 0., 0.
        return lag_peak, lag_cen, r_peak, r_cen
    
    def Weight(self):
        """
        Calculate the weights for lag posterier with ACF of contiuum LC and overlaping points following Grier+19.
        This function returns the profiles of ACF, overlapping points, and final convoloved weighting.
        """
        # calculate ACF for continuum LC
        lag_ACF, r_ACF, npt_ACF = self.ICCF_CARMA(self.t1, self.y1, self.e1, self.t1, self.y1, self.e1, tau_min = self.tau_min, 
                                              tau_max = self.tau_max, step = self.step, interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, 
                                              p1 = self.p1, q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        
        # calculate overlapping points see Grier+ 2019 P = (N(tau)/N(0))^2
        npt0 = float(self.npt[(np.abs(self.lag-0.)).argmin()])
        overlap_npt = ((self.npt/npt0))**2 # here a little different from Grier, they use npt_12
        overlap_npt = overlap_npt/overlap_npt.max() # normalized to 1
        # convolve ACF and overlapping points
        weight_unnorm = np.convolve(r_ACF, overlap_npt, 'same')
        weighting = weight_unnorm/weight_unnorm.max()
        # reset the negtive values to 0
        weighting = np.clip(weighting, 0, np.inf)
        
        return lag_ACF, r_ACF, overlap_npt, weighting   
    
    

    def MonteCarlo(self, MC_ntrials):
        """
        Run Monte Carlo for error estimation with FR/RSS method. 
        """ 
        if self.MP  ==  False:
            # no multiprocess
            lag_peak_MC = []
            lag_cen_MC = []
            r_peak_MC = []

            for n in range(MC_ntrials):
                if self.interp == 'linear':
                    if self.FR_RSS == 0:
                        # both FR and RSS
                        # RSS
                        id1 = np.hstack((np.random.randint(0, len(self.t1), len(self.t1)), [0,len(self.t1)-1]))
                        un1, ct1 = np.unique(id1, return_counts = True)
                        id2 = np.hstack((np.random.randint(0, len(self.t2), len(self.t2)), [0,len(self.t2)-1]))
                        un2, ct2 = np.unique(id2, return_counts = True)
                        # FR
                        y1 = np.random.normal(self.y1[un1], self.e1[un1]/np.sqrt(ct1))
                        y2 = np.random.normal(self.y2[un2], self.e2[un2]/np.sqrt(ct2))
                        t1, e1 = self.t1[un1], self.e1[un1]/np.sqrt(ct1)
                        t2, e2 = self.t2[un2], self.e2[un2]/np.sqrt(ct2)

                    if self.FR_RSS == 1:
                        # only FR
                        y1 = np.random.normal(self.y1, self.e1)
                        y2 = np.random.normal(self.y2, self.e2)
                        t1, t2, e1, e2 = self.t1, self.t2, self.e1, self.e2

                    if self.FR_RSS == 2:
                        # only RSS
                        id1 = np.hstack((np.random.randint(0, len(self.t1), len(self.t1)), [0,len(self.t1)-1]))
                        un1, ct1 = np.unique(id1, return_counts = True)
                        id2 = np.hstack((np.random.randint(0, len(self.t2), len(self.t2)), [0,len(self.t2)-1]))
                        un2, ct2 = np.unique(id2, return_counts = True)
                        t1, y1, e1 = self.t1[un1], self.y1[un1], self.e1[un1]/np.sqrt(ct1)
                        t2, y2, e2 = self.t2[un2], self.y2[un2], self.e2[un2]/np.sqrt(ct2)
                        
                if self.interp == 'carma':
                    # only FR for carma
                    t1_tmp = np.linspace(self.t1.min(), self.t1.max(), 1000)
                    t2_tmp = np.linspace(self.t2.min(), self.t2.max(), 1000)
                    y1_tmp = self.sample1.simulate(t1_tmp, bestfit = 'map')
                    y2_tmp = self.sample2.simulate(t2_tmp, bestfit = 'map')
                    y1 = np.random.normal(np.interp(self.t1, t1_tmp, y1_tmp), self.e1)
                    y2 = np.random.normal(np.interp(self.t2, t2_tmp, y2_tmp), self.e2)
                    t1, t2, e1, e2 = self.t1, self.t2, self.e1, self.e2
                    
                # do CCF for each MC
                lag, r, npt = self.ICCF_CARMA(t1, y1, e1, t2, y2, e2, tau_min = self.tau_min, tau_max = self.tau_max, step = self.step, interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
                # find the peak and centroid
                lag_peak, lag_cen, r_peak, r_cen = self.Lag_center(lag, r)
                #
                lag_peak_MC.append(lag_peak)
                lag_cen_MC.append(lag_cen)
                r_peak_MC.append(r_peak)
                # we do not need r_cen_MC
                
        else:
            # multiprocessing
            global loop, lock, manager, lag_peak_a, lag_cen_a, r_peak_a
            loop = 1 # call job1
            manager = Manager()
            lag_peak_a = manager.list([])
            lag_cen_a = manager.list([])
            r_peak_a = manager.list([])
            lock = Lock() # make sure the right order for appending
            p1 = Pool() # use all cores
            n = range(MC_ntrials)
            p1.map(self, n)
            p1.close()
            p1.join()
            lag_peak_MC = list(lag_peak_a)
            lag_cen_MC = list(lag_cen_a)
            r_peak_MC = list(r_peak_a)
        
        lag_peak_MC = np.array(lag_peak_MC)
        lag_cen_MC = np.array(lag_cen_MC)
        r_peak_MC = np.array(r_peak_MC)
        
        # remove failed CCF results
        ind = np.where( (lag_peak_MC < self.tau_max) & (lag_peak_MC> self.tau_min) & (r_peak_MC > self.sigmode) & (lag_cen_MC!= -9999.), True, False)
        self.failed_CCF_MC = len(lag_peak_MC)-np.sum(ind)
        
        #----------------------------------------------------------------------------
        # calculate the centers and 1 sigma error for weighted/unweighted cases.
        if np.sum(ind)>0:
            if self.weight == False:
                lag_peak_low = np.round(np.percentile(lag_peak_MC[ind], 50)-np.percentile(lag_peak_MC[ind], 16), 2)
                lag_peak_high = np.round(np.percentile(lag_peak_MC[ind], 84)-np.percentile(lag_peak_MC[ind], 50), 2)
                lag_peak = np.round(np.percentile(lag_peak_MC[ind], 50), 2)
                peak_pack_MC = np.array([lag_peak_low, lag_peak, lag_peak_high])

                lag_cen_low = np.round(np.percentile(lag_cen_MC[ind], 50)-np.percentile(lag_cen_MC[ind], 16), 2)
                lag_cen_high = np.round(np.percentile(lag_cen_MC[ind], 84)-np.percentile(lag_cen_MC[ind], 50), 2)
                lag_cen = np.round(np.percentile(lag_cen_MC[ind], 50), 2)
                cen_pack_MC = np.array([lag_cen_low, lag_cen, lag_cen_high])
            else:
                # find weighted peak distribution
                wgts = np.interp(lag_peak_MC[ind], self.lag, self.weighting)
                hist = np.histogram(lag_peak_MC[ind], bins = 50, weights = wgts)
                
                # smooth the weighted posterier histogram to find the local minimum and primary peak
                if np.sum(wgts) == 0:
                    wgts = np.ones(len(wgts))
                    print("weights is not applied due to the zero weights!")
                kde = sst.gaussian_kde(lag_peak_MC[ind], weights = wgts)
                kde.set_bandwidth(bw_method = kde.factor * self.nsmooth_wgts)
                tt = np.linspace(hist[1].min(),hist[1].max(),1000)
                self.lag_peak_kde = tt
                peak_kde = kde.evaluate(tt)
                self.peak_kde = peak_kde
                lag_peak_wgts = tt[np.argmax(peak_kde)]
               
                
                # find every local minimum in 1D array
                loc_min = peak_kde[np.r_[True, peak_kde[1:] < peak_kde[:-1]] & np.r_[peak_kde[:-1] < peak_kde[1:], True]]
                loc_min_ind = np.r_[True, peak_kde[1:] < peak_kde[:-1]] & np.r_[peak_kde[:-1] < peak_kde[1:], True]
                # find the nearest left and right minimum around peak
                try:
                    self.peak_left = np.max(tt[loc_min_ind][np.where(tt[loc_min_ind]<lag_peak_wgts)])
                except:
                    self.peak_left = self.tau_min
                try:
                    self.peak_right = np.min(tt[loc_min_ind][np.where(tt[loc_min_ind]>lag_peak_wgts)])
                except:
                    self.peak_right = self.tau_max
                region = np.where( (lag_peak_MC[ind] >self.peak_left) & (lag_peak_MC[ind]<self.peak_right), True, False)
                lag_peak_low = np.round(np.percentile(lag_peak_MC[ind][region], 50)-np.percentile(lag_peak_MC[ind][region], 16), 2)
                lag_peak_high = np.round(np.percentile(lag_peak_MC[ind][region], 84)-np.percentile(lag_peak_MC[ind][region], 50), 2)
                lag_peak = np.round(np.percentile(lag_peak_MC[ind][region], 50), 2)
                peak_pack_MC = np.array([lag_peak_low, lag_peak, lag_peak_high])
                self.lag_peak_MC_weighted = lag_peak_MC[ind][region]
                
                
                
                #------------------------------------------------------------------------
                # find weighted cen distribution
                wgts = np.interp(lag_cen_MC[ind], self.lag, self.weighting)
                hist = np.histogram(lag_cen_MC[ind], bins = 50, weights = wgts)

                # smooth the weighted posterier histogram
                if np.sum(wgts) == 0:
                    wgts = np.ones(len(wgts))
                    print("weights is not applied due to the zero weights!")
                kde = sst.gaussian_kde(lag_cen_MC[ind], weights = wgts)
                kde.set_bandwidth(bw_method = kde.factor * self.nsmooth_wgts)
                tt = np.linspace(hist[1].min(),hist[1].max(),1000)
                self.lag_cen_kde = tt
                cen_kde = kde.evaluate(tt)
                self.cen_kde = cen_kde
                lag_cen_wgts = tt[np.argmax(cen_kde)]
                
                
                #find every local minimum in 1D array
                loc_min = cen_kde[np.r_[True, cen_kde[1:] < cen_kde[:-1]] & np.r_[cen_kde[:-1] < cen_kde[1:], True]]
                loc_min_ind = np.r_[True, cen_kde[1:] < cen_kde[:-1]] & np.r_[cen_kde[:-1] < cen_kde[1:], True]
                
                
                try:
                    self.cen_left = np.max(tt[loc_min_ind][np.where(tt[loc_min_ind]<lag_cen_wgts)])
                except:
                    self.cen_left = self.tau_min
                try:
                    self.cen_right = np.min(tt[loc_min_ind][np.where(tt[loc_min_ind]>lag_cen_wgts)])
                except:
                    self.cen_right = self.tau_max
                
                region = np.where( (lag_cen_MC[ind] >self.cen_left) & (lag_cen_MC[ind]<self.cen_right), True, False)
                lag_cen_low = np.round(np.percentile(lag_cen_MC[ind][region], 50)-np.percentile(lag_cen_MC[ind][region], 16), 2)
                lag_cen_high = np.round(np.percentile(lag_cen_MC[ind][region], 84)-np.percentile(lag_cen_MC[ind][region], 50), 2)
                lag_cen = np.round(np.percentile(lag_cen_MC[ind][region], 50), 2)
                cen_pack_MC = np.array([lag_cen_low, lag_cen, lag_cen_high])
                self.lag_cen_MC_weighted = lag_cen_MC[ind][region]
                
                
            return peak_pack_MC, cen_pack_MC, lag_peak_MC[ind], lag_cen_MC[ind]
        else:
            return np.array([0., 0., 0.]), np.array([0., 0., 0.]), np.array([]), np.array([])


    def weighted_percentile(self, data, percents, wgts = None):
        ''' calculate the weighted percentile for lag posterior. This function is not used in this code, but could be useful if you want to directly apply weight on the lag posterior.
        '''
        if wgts is None:
            return np.percentile(data, percents)
        ind = np.argsort(data)
        d = data[ind]
        w = wgts[ind]
        p = 1.*w.cumsum()/w.sum()*100
        y = np.interp(percents, p, d)
        return y

    def __call__(self, n):
        #make job function callable in python2
        if loop == 1:
            return self._job1(n)
        if loop == 2:
            return self._job2(n)
        if loop == 3:
            return self._job3(n)
        if loop == 4:
            return self._job4(n)
            
    def _job1(self, n):
        # multiprocess job function for MC
        np.random.seed() # make it random
        if self.interp == 'linear':
            if self.FR_RSS == 0:
                # both FR and RSS
                # RSS
                id1 = np.hstack(( np.random.randint(0, len(self.t1), len(self.t1)),[0,len(self.t1)-1]))
                un1, ct1 = np.unique(id1, return_counts = True)
                id2 = np.hstack(( np.random.randint(0, len(self.t2), len(self.t2)),[0,len(self.t2)-1]))
                un2, ct2 = np.unique(id2, return_counts = True)
                #FR
                y1 = np.random.normal(self.y1[un1], self.e1[un1]/np.sqrt(ct1))
                y2 = np.random.normal(self.y2[un2], self.e2[un2]/np.sqrt(ct2))
                t1, e1 = self.t1[un1], self.e1[un1]/np.sqrt(ct1)
                t2, e2 = self.t2[un2], self.e2[un2]/np.sqrt(ct2)

            if self.FR_RSS == 1:
                # only FR
                y1 = np.random.normal(self.y1, self.e1)
                y2 = np.random.normal(self.y2, self.e2)
                t1, t2, e1, e2 = self.t1, self.t2, self.e1, self.e2

            if self.FR_RSS == 2:
                #only RSS
                id1 = np.hstack(( np.random.randint(0, len(self.t1), len(self.t1)),[0,len(self.t1-1)]))
                un1, ct1 = np.unique(id1, return_counts = True)
                id2 = np.hstack(( np.random.randint(0, len(self.t2), len(self.t2)),[0,len(self.t2-1)]))
                un2, ct2 = np.unique(id2, return_counts = True)
                t1, y1, e1 = self.t1[un1], self.y1[un1], self.e1[un1]/np.sqrt(ct1)
                t2, y2, e2 = self.t2[un2], self.y2[un2], self.e2[un2]/np.sqrt(ct2)
                
        if self.interp == 'carma':
                t1_tmp = np.linspace(self.t1.min(), self.t1.max(), 1000)
                t2_tmp = np.linspace(self.t2.min(), self.t2.max(), 1000)
                y1_tmp = self.sample1.simulate(t1_tmp, bestfit = 'map')
                y2_tmp = self.sample2.simulate(t2_tmp, bestfit = 'map')
                y1 = np.random.normal(np.interp(self.t1, t1_tmp, y1_tmp), self.e1)
                y2 = np.random.normal(np.interp(self.t2, t2_tmp, y2_tmp), self.e2)
                t1, t2, e1, e2 = self.t1, self.t2, self.e1, self.e2

        lag, r, npt = self.ICCF_CARMA(t1, y1, e1, t2, y2, e2, tau_min = self.tau_min, tau_max = self.tau_max, step = self.step, interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        lag_peak, lag_cen, r_peak, r_cen = self.Lag_center(lag, r)
        
        lock.acquire()
        lag_peak_a.append(lag_peak)
        lag_cen_a.append(lag_cen)
        r_peak_a.append(r_peak)
        lock.release()
        
   
    def _job2(self, nn2_ysim_t2_cadence):
        np.random.seed()
        n, ysim_t2_cadence = nn2_ysim_t2_cadence
        # multiprocess job function for CalSig, simulated y2
        lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, self.y1, self.e1, self.t2, ysim_t2_cadence, self.e2, tau_min = self.tau_min, 
                                               tau_max = self.tau_max, step = self.step, 
                                               interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                               q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        lock.acquire()
        nn2.append(n)
        lag_sim2_a.append(lagsim)
        r_sim2_a.append(rsim)
        lock.release()
    
    def _job3(self, nn1_ysim_t1_cadence):
        np.random.seed()
        n, ysim_t1_cadence = nn1_ysim_t1_cadence
        # multiprocess job function for CalSig, simulated y1
        lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, ysim_t1_cadence, self.e1, self.t2, self.y2, self.e2, tau_min = self.tau_min, 
                                               tau_max = self.tau_max, step = self.step, 
                                               interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                               q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        lock.acquire()
        nn1.append(n)
        lag_sim1_a.append(lagsim)
        r_sim1_a.append(rsim)
        lock.release()
    
    def _job4(self, nn0_ysim_t12_cadence):
        np.random.seed()
       
        n, ysim_t1_cadence, ysim_t2_cadence = nn0_ysim_t12_cadence
        # multiprocess job function for CalSig, simulated y1 and y2
        lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, ysim_t1_cadence, self.e1, self.t2, ysim_t2_cadence, self.e2, tau_min = self.tau_min, 
                                               tau_max = self.tau_max, step = self.step, 
                                               interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                               q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        
        lock.acquire()
        nn0.append(n)
        lag_sim0_a.append(lagsim)
        r_sim0_a.append(rsim)
        lock.release()

        
        

    def CalSig(self, sim_ntrials):
        """
        calculate the CCF between real (simulated) continuum LC and simulated (real) emission-line LC. The simulated continuum/emission-line LCs are produced according to the feature (i.e., p and q) of orignial LC. If interp = 'linear', we use DRW model to produce the simulated LCs.
        """
        
        # prepare for simulated y2   
        ysim_t2_cadence = []
        if self.Nmodel == 2:
            #one model to produce all simulated LCs
            if self.interp == 'linear' or self.imode == 2:
                with quiet():
                    model = cm.CarmaModel(self.t2, self.y2, self.e2, p = self.p2, q = self.q2)
                    self.sample2 = model.run_mcmc(self.mcmc_nsamples)
                    
        if self.sim_mode == 1:
            # produce sim_ntrials simulated LC for both continuum and emission lines
            sim_ntrials = sim_ntrials * 2 
        
        # produce 100x long LC and randomly select a segment from 10 to 100 and impose the same cadence and errors
        n1 = 0
        acceptance_limit = 0.001
        acceptance_rate_y2 = []
        for i in range(sim_ntrials/2):
            #print i
            nn = np.random.randint(10, 100)
            tsim = np.arange(nn*self.t2.max(), nn*self.t2.max()+self.t2.max()-self.t2.min(), 
                                (self.t2.max()-self.t2.min())/self.t2.shape[0])
            if self.Nmodel>2:
                if (i%(sim_ntrials/self.Nmodel) == 0) and (n1<self.Nmodel/2):
                    n1 = n1+1
                    with quiet():
                        model = cm.CarmaModel(self.t2, self.y2, self.e2, p = self.p2, q = self.q2)
                        sample = model.run_mcmc(self.mcmc_nsamples)  
                ysim = sample.simulate(tsim, bestfit = 'map')
                Tsim = tsim-tsim.min()+self.t2.min()
                ysim_t2 = np.interp(self.t2, Tsim, ysim)
                # let variability amp is between + - 20%  of orignial LC
                nt = 0
                good = 0. 
                bad = 0.
                while nt <1:
                    if  (good/(good+bad) < acceptance_limit) and (good + bad > 1./acceptance_limit):
                        raise Exception("Your acceptance rate is < 1% for producing mock light curve, you can set a higher MCMC_nsamples or loose the variability range (sim_var_range).")
                    if (np.random.normal(ysim_t2, self.e2).std() < self.y2.std()*(1.+self.sim_var_range)) and (np.random.normal(ysim_t2,self.e2).std() > self.y2.std()*(1.-self.sim_var_range)):
                        nt = 1
                        good = good + 1
                    else:
                        ysim = sample.simulate(tsim, bestfit = 'map')
                        ysim_t2 = np.interp(self.t2, Tsim, ysim)
                        bad = bad +1
            else:
                ysim = self.sample2.simulate(tsim, bestfit = 'map')
                Tsim = tsim-tsim.min()+self.t2.min()
                ysim_t2 = np.interp(self.t2, Tsim, ysim)
                nt=0
                good = 0.
                bad = 0.
                while nt <1:
                    if  (good/(good + bad + 1.) < acceptance_limit) and (good + bad > 1./acceptance_limit):
                        raise Exception("Your acceptance rate is < 0.1% for producing mock light curve, you can set a higher MCMC_nsamples or loose the variability range (sim_var_range).")
                    if (np.random.normal(ysim_t2, self.e2).std() < self.y2.std()*(1.+self.sim_var_range)) and (np.random.normal(ysim_t2,self.e2).std() > self.y2.std()*(1.-self.sim_var_range)):
                        nt = 1
                        good = good + 1
                    else:
                        ysim = self.sample2.simulate(tsim, bestfit = 'map')
                        ysim_t2 = np.interp(self.t2, Tsim, ysim)
                        bad = bad + 1
            acceptance_rate_y2.append(good/(good+bad+1.))
            y2_sim = ysim_t2   
            ysim_t2_cadence.append(y2_sim)
        self.acceptance_rate_y2 = np.array(acceptance_rate_y2).mean()
        ysim_t2_cadence = np.array(ysim_t2_cadence).reshape(-1, len(self.t2))
        self.ysim_t2_cadence = np.array(ysim_t2_cadence)
        
        # prepare for simulated y1
        ysim_t1_cadence = []
        if self.Nmodel == 2:
            if self.interp == 'linear' or self.imode  == 1:
                with quiet():
                    model = cm.CarmaModel(self.t1, self.y1, self.e1, p = self.p1, q = self.q1)
                    self.sample1 = model.run_mcmc(self.mcmc_nsamples)
                    
        n2 = 0 
        acceptance_rate_y1 = []
        for j in range(sim_ntrials/2):
            #print j
            nn = np.random.randint(10, 100)
            tsim = np.arange(nn*self.t1.max(), nn*self.t1.max()+self.t1.max()-self.t1.min(), 
                                (self.t1.max()-self.t1.min())/self.t1.shape[0])
            if self.Nmodel>2:
                if (j%(sim_ntrials/self.Nmodel) == 0) and (n2<self.Nmodel/2):
                    n2 = n2+1
                    with quiet():
                        model = cm.CarmaModel(self.t1, self.y1, self.e1, p = self.p1, q = self.q1)
                        sample = model.run_mcmc(self.mcmc_nsamples)
                ysim = sample.simulate(tsim, bestfit = 'map')
                Tsim = tsim-tsim.min()+self.t1.min()
                ysim_t1 = np.interp(self.t1, Tsim, ysim)
                # let variability amp is between + - 20%  of orignial LC
                nt = 0
                good = 0.
                bad = 0.
                while nt <1:
                    if  (good/(good + bad +1.) < acceptance_limit) and (good + bad > 1./acceptance_limit):
                        raise Exception("Your acceptance rate is < 1% for producing mock light curve, you can set a higher MCMC_nsamples or loose the variability range (sim_var_range).")
                    if (np.random.normal(ysim_t1, self.e1).std() < self.y1.std()*(1.+self.sim_var_range)) and (np.random.normal(ysim_t1,self.e1).std() > self.y1.std()*(1.-self.sim_var_range)):
                        nt=1
                        good = good +1
                    else:
                        ysim = sample.simulate(tsim, bestfit = 'map')
                        ysim_t1 = np.interp(self.t1, Tsim, ysim)
                        bad = bad +1
            else:
                ysim = self.sample1.simulate(tsim, bestfit = 'map')
                Tsim = tsim-tsim.min()+self.t1.min()
                ysim_t1 = np.interp(self.t1, Tsim, ysim)
                nt=0
                good = 0
                bad = 0
                while nt <1:
                    if  (good/(good + bad + 1.) < acceptance_limit) and (good + bad > 1./acceptance_limit):
                        raise Exception("Your acceptance rate is < 1% for producing mock light curve, you can set a higher MCMC_nsamples or loose the variability range (sim_var_range).")
                    if (np.random.normal(ysim_t1, self.e1).std() < self.y1.std()*(1.+self.sim_var_range)) and (np.random.normal(ysim_t1,self.e1).std() > self.y1.std()*(1.-self.sim_var_range)):
                        nt=1
                        good = good +1
                    else:
                        ysim = self.sample1.simulate(tsim, bestfit = 'map')
                        ysim_t1 = np.interp(self.t1, Tsim, ysim)
                        bad = bad +1
            acceptance_rate_y1.append(good/(good+bad+1.))
            y1_sim = ysim_t1
            ysim_t1_cadence.append(y1_sim)
   
        self.acceptance_rate_y1 = np.array(acceptance_rate_y1).mean()  
        ysim_t1_cadence = np.array(ysim_t1_cadence).reshape(-1, len(self.t1))
        self.ysim_t1_cadence = np.array(ysim_t1_cadence)
        
        #-------multiprocessing or not-----------------                        
        if self.MP == False: 
            #-----single simulated LC or both-----
            if self.sim_mode == 0:
                #for simulated y2
                lag_sim2_all = []
                r_sim2_all = []

                # measure the CCF for the simulated and real LCs
                for i in range(len(ysim_t2_cadence)):
                    lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, self.y1, self.e1, self.t2, ysim_t2_cadence[i], self.e2, tau_min = self.tau_min, 
                                                           tau_max = self.tau_max, step = self.step, 
                                                           interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                                           q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
                    lag_sim2_all.append(lagsim)
                    r_sim2_all.append(rsim)

                lag_sim2_all = np.array(lag_sim2_all)
                r_sim2_all = np.array(r_sim2_all)

                #for simulated y1
                lag_sim1_all = []
                r_sim1_all = []
                plt.figure(figsize = (8, 6))
                for j in range(len(ysim_t1_cadence)):

                    lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, ysim_t1_cadence[j], self.e1, self.t2, self.y2, self.e2, tau_min = self.tau_min, 
                                                           tau_max = self.tau_max, step = self.step, 
                                                           interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                                           q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
                    lag_sim1_all.append(lagsim)
                    r_sim1_all.append(rsim)

                lag_sim1_all = np.array(lag_sim1_all)
                r_sim1_all = np.array(r_sim1_all)
            else:
                lag_sim0_all = []
                r_sim0_all = []

                # measure the CCF for the simulated and real LCs
                for i in range(len(ysim_t1_cadence)):
                    lagsim, rsim, nptsim = self.ICCF_CARMA(self.t1, self.y1, ysim_t1_cadence[i], self.t2, ysim_t2_cadence[i], self.e2, tau_min = self.tau_min, 
                                                           tau_max = self.tau_max, step = self.step, 
                                                           interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, 
                                                           q1 = self.q1, p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
                    lag_sim0_all.append(lagsim)
                    r_sim0_all.append(rsim)

                lag_sim0_all = np.array(lag_sim0_all)
                r_sim0_all = np.array(r_sim0_all)
                
        else:
            #multiprocessing
            global loop, lag_sim1_a, lag_sim2_a, r_sim1_a, r_sim2_a, nn1, nn2, nn0, lag_sim0_a, r_sim0_a
            loop = 2 # go job2
            lag_sim1_a = manager.list([])
            lag_sim2_a = manager.list([])
            r_sim1_a = manager.list([])
            r_sim2_a = manager.list([])
            nn1 = manager.list([])
            nn2 = manager.list([])
            nn0 = manager.list([])
            lag_sim0_a = manager.list([])
            r_sim0_a = manager.list([])
            p2 = Pool()
            
            p2.map(self, zip(range(len(ysim_t2_cadence)), ysim_t2_cadence))
            p2.close()
            p2.join()
            lag_sim2_all = np.array(list(lag_sim2_a))[np.argsort(nn2)]
            r_sim2_all = np.array(list(r_sim2_a))[np.argsort(nn2)]
            
            loop = 3 # go job3
            p3 = Pool()
            p3.map(self, zip(range(len(ysim_t2_cadence)), ysim_t1_cadence))
            p3.close()
            p3.join()
            lag_sim1_all = np.array(list(lag_sim1_a))[np.argsort(nn1)]
            r_sim1_all = np.array(list(r_sim1_a))[np.argsort(nn1)]
            
            loop =4 # go job4
            p4 = Pool()
            p4.map(self, zip(range(len(ysim_t1_cadence)), ysim_t1_cadence, ysim_t2_cadence))
            p4.close()
            p4.join()
            lag_sim0_all = np.array(list(lag_sim0_a))[np.argsort(nn0)]
            r_sim0_all = np.array(list(r_sim0_a))[np.argsort(nn0)]
           
        
        # save all the simulated CCF, 1 means simulated y1, 2 means simulated y2
        if self.sim_mode ==0:
            lag_sim = np.array(np.vstack((lag_sim1_all, lag_sim2_all)))
            r_sim = np.array(np.vstack((r_sim1_all, r_sim2_all)))
        else:
            lag_sim = lag_sim0_all
            r_sim = r_sim0_all
        
        #--------------------------------------------------------------------------
        #calculate significance p_all, p_positive, p_peak
        lag_sim_peak = []
        lag_sim_cen = []
        r_sim_peak = []
        r_sim_cen = []
        
        if sim_ntrials >0: 
            for i in range(r_sim.shape[0]):
                lag_sim_peak.append(self.Lag_center(lag_sim[i, :], r_sim[i, :])[0])
                lag_sim_cen.append(self.Lag_center(lag_sim[i, :], r_sim[i, :])[1])
                r_sim_peak.append(self.Lag_center(lag_sim[i, :], r_sim[i, :])[2])
                r_sim_cen.append(self.Lag_center(lag_sim[i, :], r_sim[i, :])[3]) 
                
            n1 = float(np.sum(np.where(r_sim_peak>self.r.max(), True, False))) # all CCF > r_max
            n2 = float(np.sum(np.where( (r_sim_peak>self.r.max()) & (np.array(lag_sim_peak)>0) , True, False))) # all CCF > r_max & lag >0
            n_total = float(len(r_sim_peak)) # total simulation       
            n_total_positive = float(np.sum(np.where(np.array(lag_sim_peak)>0, True, False))) # lag>0
            p_peak = float(np.sum(np.where(r_sim[:, np.argmax(self.r)]>self.r.max(), True, False)))/n_total
            try:
                p_positive = np.round(n2/n_total_positive, 5)
            except:
                p_positive = -1.
            p_all = np.round(n1/n_total, 5)
        else:
            p_all = -1.
            p_positve = -1.
        self.p_peak = p_peak
        self.p_all = p_all
        self.p_positive = p_positive
        self.lag_sim_peak = np.array(lag_sim_peak)
        self.lag_sim_cen = np.array(lag_sim_cen)
        self.r_sim_peak = np.array(r_sim_peak)
        self.r_sim_cen = np.array(r_sim_cen)
        return lag_sim, r_sim
    
    def CalSig_Other_Peak(self, low1, high1, low2 = None, high2 = None):
        '''
        calculate the significance of other peaks
        low1, high1: to search a secondary peak in [low1, high1]
        low2, high2: to calculate the significance in [low2, high2]
        '''
        try:
            if low2 == None:
                low2 = 0
            if high2 == None:
                high2 = self.tau_max
                
            ind = np.where( (self.lag>low1) & (self.lag<high1), True, False)
            lag = np.argmax(self.r[ind].max())
            r = self.r[ind].max()

            lag_peak = []
            r_peak = []
            for i in range(self.r_sim.shape[0]):
                lag_peak.append(self.lag_sim[i,:][np.argmax(self.r_sim[i,:])])
                r_peak.append(np.max(self.r_sim[i,:]))
            lag_peak=np.array(lag_peak)
            r_peak=np.array(r_peak)

            idx = np.where((lag_peak > low2) & (lag_peak < high2), True, False)
            idx2 = np.where((lag_peak > low2) & (lag_peak < high2) & (r > r_peak), True, False)

            return 1. - float(np.sum(idx2))/float(np.sum(idx))
        except:
            raise Exception("nsim_trials must be >= 2!")
    
    def PlotLC(self):
        # plot the orignial/detrened light curve and their CARMA model used for interpolation.
        plt.figure(figsize = (15, 10))
        plt.subplots_adjust(hspace=0)
        # plot contimuum ------------------------------
        try:
            if self.shift == 'peak':
                offset = self.peak_pack_MC[1]
            else:
                offset = self.cen_pack_MC[1]
        except:
            if self.shift == 'peak':
                offset = self.lag_peak
            else:
                if self.lag_cen != -9999.:
                    offset = self.lag_cen
                else:
                    offset = self.lag_peak
        xmin,xmax = min(self.t1.min(),self.t2.min()-offset), max(self.t1.max(),self.t2.max()-offset)
        ax = plt.subplot(311)
        if self.interp == 'carma':
            plt.errorbar(self.t1, self.y1, yerr = self.e1, fmt = '.', color = 'k', lw = 1)    
            time = np.linspace(self.t1.min(), self.t1.max(), 1000)
            try:
                if self.carma_model == 'mean':
                    model_mean, model_var = self.sample1.predict(time, bestfit = 'map')
                    plt.plot(time, model_mean, '-b', lw = 1)
                    low = model_mean - np.sqrt(model_var)
                    high = model_mean + np.sqrt(model_var)
                    plt.fill_between(time, low, high, facecolor = 'blue', alpha = 0.2)
                else:
                    ysim = self.sample1.simulate(time, bestfit = 'map')
                    plt.plot(time, ysim, 'b-')
            except:
                pass
            plt.title(self.name+':  interp = '+self.interp+', CarmaModel = '+str(self.carma_model)+', p1 = '+str(self.p1)+', q1 = '+str(self.q1)+', p2 = '+str(self.p2)+', q2 = '+str(self.q2)+', detrend = '+str(self.detrend) )
        else:
            plt.errorbar(self.t1, self.y1, yerr = self.e1, fmt = '.', color = 'k', lw = 1)
            plt.errorbar(self.t1, self.y1, yerr = self.e1, fmt = '-', color = 'k', lw=1, alpha=0.2)
            plt.title(self.name+':  interp = '+self.interp+', detrend = '+ str(self.detrend))
        if self.detrend > 0:
            time = np.linspace(self.t1.min(),self.t1.max(),1000)
            cont = np.poly1d(np.polyfit(self.t1, self.y1, self.detrend))
            plt.plot(time,cont(time),'grey')
        plt.text(0.8, 0.9, r'$\rm VAR_{SNR}$ = '+str(np.round(self.var_sn1, 1)), fontsize=16, transform = ax.transAxes)
        plt.ylabel('Flux')
        plt.xticks([])
        plt.xlim(xmin,xmax)
        # plot emission line ----------------------------
        ax = plt.subplot(312)
        if self.interp == 'carma':
            plt.errorbar(self.t2, self.y2, yerr = self.e2, fmt = '.', color = 'k', lw = 1)    
            time = np.linspace(self.t2.min(), self.t2.max(), 1000)
            try:
                if self.carma_model == 'mean':
                    model_mean, model_var = self.sample2.predict(time, bestfit = 'map')
                    plt.plot(time, model_mean, '-b', lw = 1)
                    low = model_mean - np.sqrt(model_var)
                    high = model_mean + np.sqrt(model_var)
                    plt.fill_between(time, low, high, facecolor = 'blue', alpha = 0.2)
                else:
                    ysim = self.sample2.simulate(time, bestfit = 'map')
                    plt.plot(time, ysim, 'b-')
            except:
                    pass
        else:
            plt.errorbar(self.t2, self.y2, yerr = self.e2, fmt = '.', color = 'k', lw = 1)
            plt.errorbar(self.t2, self.y2, yerr = self.e2, fmt = '-', color = 'k', lw = 1, alpha = 0.1)
        if self.detrend > 0:
            time = np.linspace(self.t2.min(),self.t2.max(),1000)
            line = np.poly1d(np.polyfit(self.t2, self.y2, self.detrend))
            plt.plot(time,line(time),'grey')
        plt.text(0.8, 0.9, r'$\rm VAR_{SNR}$ = '+str(np.round(self.var_sn2, 1)), fontsize=16, transform = ax.transAxes)
        plt.ylabel('Flux')
        plt.xticks([])
        plt.xlim(xmin,xmax) 
        
        # plot normalized LCs--------------------------
        ax = plt.subplot(313)
        if self.detrend > 0:
            # use new t1,t2,y1,y2 but orignial e1 e2
            t1, y1, t2, y2 = self.Detrend(self.t1, self.y1, self.t2, self.y2)
            plt.errorbar(t1, (y1-y1.mean())*y2.std()/y1.std(), yerr = self.e1, fmt = '.', color = 'b', label = 'cont' )
            plt.errorbar(t2-offset, y2-y2.mean() , yerr = self.e2, fmt = '.', color = 'r', label = 'line')
        else:
            plt.errorbar(self.t1, (self.y1-self.y1.mean())*self.y2.std()/self.y1.std(), yerr = self.e1, fmt = '.', color = 'b', label = 'cont' )
            plt.errorbar(self.t2-offset, self.y2-self.y2.mean() , yerr = self.e2, fmt = '.', color = 'r', label = 'line')
        plt.text(0.8, 0.1, 'offset = '+ str(offset), transform = ax.transAxes, fontsize = 16)
        plt.ylabel('Flux')
        plt.xlabel('Time')
        plt.xlim(xmin,xmax)
        plt.legend(fontsize=16)
        plt.tight_layout()
        
        if self.save == True:
            plt.savefig(self.path+self.name+'_LC.pdf')
        
    def PlotCCF(self): 
        # Plot CCF results
        if self.MC_ntrials<1:
            # only plot the CCF panel
            plt.figure(figsize = (8, 6))
            ax = plt.subplot(111)
            plt.plot(self.lag, self.r)
            idx = np.where( (self.lag<self.right) & (self.lag>self.left), True, False)
            plt.plot(self.lag[idx], self.r[idx], 'r')
            if self.imode!= 1:
                plt.plot(self.lag_2, self.r_2, 'grey', alpha = 0.2)
            if self.imode!= 2:
                plt.plot(self.lag_1, self.r_1, 'c', alpha = 0.2)
            plt.xlabel('Lag (days)')
            plt.ylabel('r')
            plt.text(0.01, 0.9, r'$\rm \tau_{peak} = $'+str(np.round(self.lag_peak, 2)), transform = ax.transAxes, fontsize = 16)
            plt.text(0.01, 0.8, r'$\rm \tau_{cent} = $'+str(np.round(self.lag_cen, 2)), transform = ax.transAxes, fontsize = 16)
            plt.tight_layout()
        else:
            if self.sim_ntrials >1: 
                plt.figure(figsize = (15, 10))
                ax = plt.subplot(2, 6, (1, 2))
            else:
                plt.figure(figsize = (15, 5))
                ax = plt.subplot(1, 6, (1, 2))
            plt.plot(self.lag, self.r)
            idx = np.where( (self.lag<self.right) & (self.lag>self.left), True, False)
            plt.plot(self.lag[idx], self.r[idx], 'r')
            if self.imode!= 1:
                plt.plot(self.lag_2, self.r_2, 'grey', alpha = 0.2)
            if self.imode!= 2:
                plt.plot(self.lag_1, self.r_1, 'c', alpha = 0.2)
            if self.weight == True:
                plt.plot(self.lag, self.weighting, 'm', alpha = 0.5, label = 'Weights')
                plt.legend(loc = 4, fontsize = 16)
            
            plt.xlabel('Lag (days)')
            plt.ylabel('r')
            plt.text(0.01, 0.9, r'$\rm \tau_{peak} = $'+str(np.round(self.lag_peak, 2)), transform = ax.transAxes, fontsize = 16)
            plt.text(0.01, 0.8, r'$\rm \tau_{cent} = $'+str(np.round(self.lag_cen, 2)), transform = ax.transAxes, fontsize = 16)
            
            # ------------------------------------------------------------------------------------------------------------------------
            # plot error panel
            # --------------lag peak--------------
            if self.sim_ntrials >1:
                ax = plt.subplot(2, 6, (3, 4))
            else:
                ax = plt.subplot(1, 6, (3, 4))
            
            plt.hist(self.lag_peak_MC, bins = 50, histtype = 'step', lw = 2, alpha = 0.8)
            
            if self.weight == True:
                weights = np.interp(self.lag_peak_MC, self.lag, self.weighting)
                hist = plt.hist(self.lag_peak_MC, weights = weights, bins = 50, color = 'g', histtype = 'step', ls = 'dashed', lw = 2, alpha = 0.5)
                plt.axvspan(self.peak_left, self.peak_right, alpha = 0.1, color = 'grey')
                fc = (hist[0].max()/self.peak_kde.max())
                idx = np.where(self.peak_kde*fc>0.01, True, False)
                plt.plot(self.lag_peak_kde[idx], self.peak_kde[idx]*fc, alpha = 0.5)

            plt.xlabel('Peak Lag (days)')
            plt.ylabel('Number')
            plt.title(self.name)
            ymin, ymax = plt.ylim()
            ymax = ymax*1.1
            
            plt.plot([self.peak_pack_MC[1], self.peak_pack_MC[1]], [ymin, ymax], 'r--')
            plt.plot([self.peak_pack_MC[1]-self.peak_pack_MC[0], self.peak_pack_MC[1]-self.peak_pack_MC[0]], [ymin, ymax], 'r:')
            plt.plot([self.peak_pack_MC[1]+self.peak_pack_MC[2], self.peak_pack_MC[1]+self.peak_pack_MC[2]], [ymin, ymax], 'r:')
            plt.text(0.01, 0.9, r'$\rm \tau_{peak} = '+str(np.round(self.peak_pack_MC[1], 2))+'^{+'+str(np.round(self.peak_pack_MC[2], 2))+'}'+'_{-'+str(np.round(self.peak_pack_MC[0], 2))+'}$', transform = ax.transAxes, fontsize = 15)
            plt.ylim(ymin, ymax)
            
            # ---------lag centroid --------------
            if self.sim_ntrials >1:
                ax = plt.subplot(2, 6, (5, 6))
            else:
                ax = plt.subplot(1, 6, (5, 6))

            plt.hist(self.lag_cen_MC, bins = 50, histtype = 'step', lw = 2, alpha = 0.8)
            if self.weight == True:
                weights = np.interp(self.lag_cen_MC, self.lag, self.weighting)
                hist = plt.hist(self.lag_cen_MC, weights = weights, bins = 50, color = 'g', histtype = 'step', ls = 'dashed', lw = 2, alpha = 0.5)
                plt.axvspan(self.cen_left, self.cen_right, alpha = 0.1, color = 'grey')
                fc = (hist[0].max()/self.cen_kde.max())
                idx = np.where(self.cen_kde*fc>0.01, True, False)
                plt.plot(self.lag_cen_kde[idx], self.cen_kde[idx]*fc, alpha = 0.5)
            plt.xlabel('Centroid Lag (days)')
            plt.ylabel('Number')
            ymin, ymax = plt.ylim()
            ymax = ymax*1.1
            plt.plot([self.cen_pack_MC[1], self.cen_pack_MC[1]], [ymin, ymax], 'r--')
            plt.plot([self.cen_pack_MC[1]-self.cen_pack_MC[0], self.cen_pack_MC[1]-self.cen_pack_MC[0]], [ymin, ymax], 'r:')
            plt.plot([self.cen_pack_MC[1]+self.cen_pack_MC[2], self.cen_pack_MC[1]+self.cen_pack_MC[2]], [ymin, ymax], 'r:')
            plt.text(0.01, 0.9, r'$\rm \tau_{cent} = '+str(np.round(self.cen_pack_MC[1], 2))+'^{+'+str(np.round(self.cen_pack_MC[2], 2))+'}'+'_{-'+str(np.round(self.cen_pack_MC[0], 2))+'}$', transform = ax.transAxes, fontsize = 15)
            plt.ylim(ymin, ymax)
            
            # -------------------------------------------------------------------------------------------------------------------------------
            # plot simulated CCF panels
            if self.sim_ntrials >1:
                ax = plt.subplot(2, 6, (7, 11))
                if self.sim_mode == 0:
                    for i in range(self.r_sim.shape[0]):
                        if i <self.r_sim.shape[0]/2.:
                            # simulated y1
                            plt.plot(self.lag_sim[i, :], self.r_sim[i, :], 'gray', lw = 1, alpha = 0.1)
                        else:
                            # simualted y2
                            plt.plot(self.lag_sim[i, :], self.r_sim[i, :], 'c', lw = 1, alpha = 0.1)
                else:
                    for i in range(self.r_sim.shape[0]):
                        plt.plot(self.lag_sim[i, :], self.r_sim[i, :], 'gray', lw = 1, alpha = 0.1)
                sig1, sig2, sig3 = [], [], []
                
                for i in range(self.r_sim.shape[1]):
                    sig1.append(np.percentile(self.r_sim[:, i], 68.3))
                    sig2.append(np.percentile(self.r_sim[:, i], 95.5))
                    sig3.append(np.percentile(self.r_sim[:, i], 99.7))
                
                plt.plot(self.lag_sim[0, :], np.array(sig1), 'g', lw = 1)
                plt.plot(self.lag_sim[0, :], np.array(sig2), 'm', lw = 1)
                plt.plot(self.lag_sim[0, :], np.array(sig3), 'b', lw = 1)
                
                # for legend
                plt.hist([], np.arange(-2, -1), histtype = 'step', label = 'peak')
                plt.hist([], np.arange(-2, -1), histtype = 'step', label = 'cent')
                plt.plot([], [], 'g', label = r'$\rm 1\sigma$')
                plt.plot([], [], 'm', label = r'$\rm 2\sigma$')
                plt.plot([], [], 'b', label = r'$\rm 3\sigma$')
                if self.sim_mode == 0:
                    plt.plot([], [], 'grey', label = r'$\rm interp\ y1$')
                    plt.plot([], [], 'c', label = r'$\rm interp\ y2$')
                else:
                    plt.plot([], [], 'grey', label = r'$\rm interp\ y1&y2$')
        
                plt.xlim(self.tau_min, self.tau_max)
                plt.ylim(-1, 1)
                plt.plot(self.lag, self.r, 'r')  
                plt.xlabel('Lag (Days)')
                plt.ylabel('r')
                plt.legend(loc = 4, framealpha = 0.8, ncol = 2)
                plt.text(0.02, 0.1, r'$\rm p_{\tau > 0} = $'+str(self.p_positive), transform = ax.transAxes, fontsize = 20)

                # ---------histogram panel--------------------------
                ax = plt.subplot(2, 6, (12))
                plt.hist(self.r_sim_peak, np.arange(-1, 1, 0.01), histtype = 'step', normed = True, orientation = "horizontal", label = 'peak')
                plt.hist(self.r_sim_cen, np.arange(-1, 1, 0.01), histtype = 'step', normed = True, orientation = "horizontal", label = 'cen')
                plt.xlabel('N')
                plt.xticks([1])
                plt.plot([plt.xlim()[0], plt.xlim()[1]], [self.r.max(), self.r.max()], 'r:')
                plt.ylim(-1, 1)
        plt.tight_layout()
        
        if self.save == True:
            plt.savefig(self.path+self.name+'_CCF.pdf')
            
    def CheckMockLC(self, nth_highest):
        "Check the mock light cruve with higher CCF peak."
        
        ind=np.where( (self.lag_sim_peak>0) & (self.r_sim_peak>self.r.max()))
        idx=np.argsort(self.r_sim_peak[ind])
        print("Totally, " + str(len(idx))+" simulations have higher CCF peaks.")
        if nth_highest > len(idx):
            raise Exception("There are only "+str(len(idx))+" CCFs hgiher than observed one!" )
        else:
            num = nth_highest
        ind= np.array(ind).flatten()[idx[-1*num]]
        
        if self.sim_mode ==0:
            if ind<self.sim_ntrials/2:
                #sim cont
                y1,y2 = self.ysim_t1_cadence[ind,:],self.y2 
                label1 = 'sim cont'
            else:
                y1,y2 = self.y1,self.ysim_t2_cadence[ind-self.sim_ntrials/2,:]
                label2 = 'sim line'
        else:
            y1,y2 = self.ysim_t1_cadence[ind,:],self.ysim_t2_cadence[ind,:]
            label1,label2 = 'sim cont', 'sim line'
            
        lag, r, _ = self.ICCF_CARMA(self.t1, y1, self.e1, self.t2, y2, self.e2, tau_min = self.tau_min, tau_max = self.tau_max, 
                    step = self.step, interp = self.interp, mcmc_nsamples = self.mcmc_nsamples, p1 = self.p1, q1 = self.q1, 
                    p2 = self.p2, q2 = self.q2, carma_model = self.carma_model, imode = self.imode)
        lag_peak, lag_cen, _, _ = self.Lag_center(lag, r)
        if lag_cen != -9999.0:
            offset = np.round(lag_cen,2)
        else:
            offset = np.round(lag_peak,2)
        plt.figure(figsize=(15,12))
        ax=plt.subplot(411)
        plt.plot(lag,r,'k')
        plt.text(0.05, 0.1, r'$r_{\rm max}$ = '+ str(np.round(r.max(),4)), transform = ax.transAxes, fontsize = 16, color = 'r')
        plt.xlabel('Lag (days)')
        plt.ylabel('r')
        
        
        plt.subplot(412)
        if label1 == 'sim cont':
            lab = label1
        else:
            lab = 'cont'
        xmin,xmax = min(self.t1.min(),self.t2.min()-offset), max(self.t1.max(),self.t2.max()-offset)
        plt.errorbar(self.t1, y1, yerr = self.e1, fmt = '.', color = 'b', lw = 1, label = lab) 
        plt.xlim(xmin,xmax)
        plt.legend(fontsize=16)
        plt.ylabel('Flux')
        if self.detrend > 0:
            time = np.linspace(self.t1.min(),self.t1.max(),1000)
            cont = np.poly1d(np.polyfit(self.t1, y1, self.detrend))
            plt.plot(time,cont(time),'grey')
        
        plt.subplot(413)
        if label2 == 'sim line':
            lab = label2
        else:
            lab = 'line'
        plt.errorbar(self.t2, y2, yerr = self.e2, fmt = '.', color = 'r', lw = 1, label = lab) 
        plt.xlim(xmin,xmax)
        plt.ylabel('Flux')
        plt.legend(fontsize=16)
        if self.detrend > 0:
            time = np.linspace(self.t2.min(),self.t2.max(),1000)
            cont = np.poly1d(np.polyfit(self.t2, y2, self.detrend))
            plt.plot(time,cont(time),'grey')
        
        ax = plt.subplot(414)
        if self.detrend > 0:
            # use new t1,t2,y1,y2 but orignial e1 e2
            t1, y1, t2, y2 = self.Detrend(self.t1, y1, self.t2, y2)
            plt.errorbar(t1, (y1-y1.mean())*y2.std()/y1.std(), yerr = self.e1, fmt = '.', color = 'b', label = 'cont' )
            plt.errorbar(t2-offset, y2-y2.mean() , yerr = self.e2, fmt = '.', color = 'r', label = 'line')
        else:
            plt.errorbar(self.t1, (y1-y1.mean())*y2.std()/y1.std(), yerr = self.e1, fmt = '.', color = 'b', label = 'cont' )
            plt.errorbar(self.t2-offset, y2-y2.mean() , yerr = self.e2, fmt = '.', color = 'r', label = 'line')
        plt.text(0.8, 0.1, 'offset = '+ str(offset), transform = ax.transAxes, fontsize = 16)
        plt.ylabel('Flux')
        plt.xlabel('Time')
        plt.xlim(xmin,xmax)
        plt.legend(fontsize=16)
        plt.tight_layout()
        
        
        
    def SaveResults(self):
        if self.sim_ntrials <2:
            self.p_positive = 0
            self.p_peak = 0
            self.p_all =0
            self.r_sim_peak = np.zeros(1)
            self.r_sim_cen = np.zeros(1)
            self.lag_sim = np.zeros(2).reshape(2,-1)
            self.r_sim = np.zeros(2).reshape(2,-1)
            self.ysim_t1_cadence = np.zeros(2).reshape(2,-1)
            self.ysim_t2_cadence = np.zeros(2).reshape(2,-1)
            
        if self.MC_ntrials >0: 
            #  ------1st extension---------
            c1 = fits.Column(name = 'Name', array = np.array([self.name]), format = '20A')
            c2 = fits.Column(name = 'interpolation', array = np.array([self.interp]), format = '20A')
            c3 = fits.Column(name = 'carma_model', array = np.array([self.carma_model]), format = '20A')
            c4 = fits.Column(name = 'p1', array = np.array([self.p1]), format = 'K')
            c5 = fits.Column(name = 'q1', array = np.array([self.q1]), format = 'K')
            c6 = fits.Column(name = 'p2', array = np.array([self.p1]), format = 'K')
            c7 = fits.Column(name = 'q2', array = np.array([self.q2]), format = 'K')
            c8 = fits.Column(name = 'detrend', array = np.array([self.detrend]), format = 'F')
            c9 = fits.Column(name = 'tau_min', array = np.array([self.tau_min]), format = 'F')
            c10 = fits.Column(name = 'tau_max', array = np.array([self.tau_max]), format = 'F')
            c11 = fits.Column(name = 'step', array = np.array([self.step]), format = 'F')
            c12 = fits.Column(name = 'sig_cut', array = np.array([self.sig_cut]), format = 'F')
            c13 = fits.Column(name = 'imode', array = np.array([self.imode]), format = 'K')
            c14 = fits.Column(name = 'MC_trials', array = np.array([self.MC_ntrials]), format = 'K')
            c15 = fits.Column(name = 'sim_trials', array = np.array([self.sim_ntrials]), format = 'K')
            c16 = fits.Column(name = 'mcmc_nsamples', array = np.array([self.mcmc_nsamples]), format = 'K')
            
            c17 = fits.Column(name = 'lag_peak', array = np.array([self.lag_peak]), format = 'F')
            c18 = fits.Column(name = 'lag_cen', array = np.array([self.lag_cen]), format = 'F')
            c19 = fits.Column(name = 'peak_mid_MC', array = np.array([self.peak_pack_MC[1]]), format = 'F')
            c20 = fits.Column(name = 'peak_low_err_MC', array = np.array([self.peak_pack_MC[0]]), format = 'F')
            c21 = fits.Column(name = 'peak_high_err_MC', array = np.array([self.peak_pack_MC[2]]), format = 'F')
            
            c22 = fits.Column(name = 'cen_mid_MC', array = np.array([self.cen_pack_MC[1]]), format = 'F')
            c23 = fits.Column(name = 'cen_low_err_MC', array = np.array([self.cen_pack_MC[0]]), format = 'F')
            c24 = fits.Column(name = 'cen_high_err_MC', array = np.array([self.cen_pack_MC[2]]), format = 'F')
            c25 = fits.Column(name = 'p_positive', array = np.array([self.p_positive]), format = 'F')
            c26 = fits.Column(name = 'p_all', array = np.array([self.p_all]), format = 'F')
            c27 = fits.Column(name = 'p_peak', array = np.array([self.p_peak]), format = 'F')
            c28 = fits.Column(name = 'failed_CCF_MC', array = np.array([self.failed_CCF_MC]), format = 'F')
            c29 = fits.Column(name = 'VAR_SN1', array = np.array([self.var_sn1]), format = 'F')
            c30 = fits.Column(name = 'VAR_SN2', array = np.array([self.var_sn2]), format = 'F')
            c31 = fits.Column(name = 'rmax', array = np.array([self.rmax]), format = 'F')
            
            #---2rd extension--------------------
            c32 = fits.Column(name = 't1', array = self.t1, format = 'F')
            c33 = fits.Column(name = 'y1', array = self.y1, format = 'F')
            c34 = fits.Column(name = 'e1', array = self.e1, format = 'F')
            c35 = fits.Column(name = 't2', array = self.t2, format = 'F')
            c36 = fits.Column(name = 'y2', array = self.y2, format = 'F')
            c37 = fits.Column(name = 'e2', array = self.e2, format = 'F')

            c38 = fits.Column(name = 'lag', array = self.lag, format = 'F')
            c39 = fits.Column(name = 'r', array = self.r, format = 'F')
            c40 = fits.Column(name = 'npt', array = self.npt, format = 'F')
            
            if self.imode !=2:
                c41 = fits.Column(name = 'r_1', array = self.r_1, format = 'F')
                c42 = fits.Column(name = 'npt_1', array = self.npt_1, format = 'F')
            else:
                c41 = fits.Column(name = 'r_1', array = np.array([]), format = 'F')
                c42 = fits.Column(name = 'npt_1', array = np.array([]), format = 'F')
            if self.imode !=1:
                c43 = fits.Column(name = 'r_2', array = self.r_2, format = 'F')
                c44 = fits.Column(name = 'npt_2', array = self.npt_2, format = 'F')
            else:
                c43 = fits.Column(name = 'r_2', array = np.array([]), format = 'F')
                c44 = fits.Column(name = 'npt_2', array = np.array([]), format = 'F')
                
            c45 = fits.Column(name = 'lag_peak_MC', array = self.lag_peak_MC, format = 'F')
            c46 = fits.Column(name = 'lag_cen_MC', array = self.lag_cen_MC, format = 'F')
            c47 = fits.Column(name = 'r_sim_peak', array = self.r_sim_peak, format = 'F')
            c48 = fits.Column(name = 'r_sim_cen', array = self.r_sim_cen, format = 'F')
            
            c49 = fits.Column(name = 'lag_sim', array = self.lag_sim, dim = '('+str(self.lag_sim.shape[1])+')', format = str(self.lag_sim.shape[1])+'E')
            c50 = fits.Column(name = 'r_sim', array = self.r_sim, dim = '('+str(self.r_sim.shape[1])+')', format = str(self.r_sim.shape[1])+'E')
            c51 = fits.Column(name = 'ysim_t1_cadence', array = self.ysim_t1_cadence, dim = '('+str(self.ysim_t1_cadence.shape[1])+')', format = str(self.ysim_t1_cadence.shape[1])+'E')
            c52 = fits.Column(name = 'ysim_t2_cadence', array = self.ysim_t2_cadence, dim = '('+str(self.ysim_t2_cadence.shape[1])+')', format = str(self.ysim_t2_cadence.shape[1])+'E')
            
            # produce fits
            ex0 = fits.PrimaryHDU()
            ex1 = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31])
            
            if self.weight == True:
                c53 = fits.Column(name = 'weighting', array = self.weighting, format = 'F')
                c54 = fits.Column(name = 'lag_ACF', array = self.lag_ACF, format = 'F')
                c55 = fits.Column(name = 'r_ACF', array = self.r_ACF, format = 'F')
                ex2 = fits.BinTableHDU.from_columns([c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55])
            else:
                ex2 = fits.BinTableHDU.from_columns([c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46])
            
            if self.lite == False:
                hdul = fits.HDUList([ex0, ex1, ex2])
            else:
                hdul = fits.HDUList([ex0, ex1])
            hdul.writeto(self.path+self.name+'.fits', overwrite = True)


