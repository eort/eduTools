import re
import os.path as op
import logging
from datetime import timedelta

import autoreject as ar
from autoreject import read_auto_reject
import numpy as np
from mne import (Annotations, read_evokeds, write_evokeds,
                 compute_proj_raw, annotations_from_events,
                 events_from_annotations, Epochs,
                 read_vectorview_selection, pick_types,
                 concatenate_epochs)
from mne.viz import (plot_events, plot_projs_topomap)
from mne.preprocessing import (ICA, find_bad_channels_maxwell,
                               maxwell_filter,
                               oversampled_temporal_projection,  
                               create_eog_epochs, create_ecg_epochs)
from mne.time_frequency import tfr_morlet
from mne.surface import write_surface
import mne_bids as mb
import pandas as pd
from pandas import read_csv

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from seaborn import heatmap

from . import plot as pt

def annotate_breaks(raw, events, ep_info):
    """Finds the breaks in the experiment and annotates them as bad """

    # read out epoch related info:
    stim_min = ep_info['stim_min']
    stim_max = ep_info['stim_max']
    trial_trg = ep_info['trial_trg']
    block_trg = ep_info['block_trg']
    bad_buffer = ep_info['bad_buffer']
    pre_time = np.abs(stim_min) + bad_buffer
    post_time = np.abs(stim_max) + bad_buffer

    # find all the start block triggers (breaks happen right before)
    start_block = np.where(events[:, 2] == block_trg)[0]

    # when did the pause end?
    end_times = events[start_block, 0] / raw.info['sfreq'] - pre_time

    # move back in the event history to find the end of the previous trial
    start_time = []
    for idx in start_block:
        while (events[idx, 2] not in trial_trg):
            idx -= 1
            # if we are at start of exp, everything before trigger can go
            if idx <= 0:
                break
        # add the times of when the pause starts
        if idx <= 0 and end_times[0] - raw.first_time > 0:
            start_time.append(raw.first_time)
        else:
            # in other blocks, start 3 seconds after last stimulus onset
            start_time.append(events[idx, 0] / raw.info['sfreq'] + post_time)

    # duration of the break
    duration = end_times - start_time

    # create Annotations
    return Annotations(start_time, duration, 'BAD_break',
                       raw.info['meas_date'])


def shift_onset_annotations(raw_annots, event, time_shift):
    # extract info
    onset = raw_annots.onset
    duration = raw_annots.duration
    description = raw_annots.description
    orig_time = raw_annots.orig_time
    # update onset of events
    bad_event_idx = np.where(raw_annots.description == event)[0]
    bad_event_idx = bad_event_idx[1:]
    onset[bad_event_idx] = onset[bad_event_idx] + time_shift
    return Annotations(onset, duration, description, orig_time)


def check_epochs(epochs):
    """
    Check wether Epochs were falsely marked as bad due to the step "EXCLUDE BAD PERIODS" which marks breaks in the experiment as bad events
    """
    epochs.load_data()
    for ep in epochs.drop_log:
        if ep: 
            raise ValueError(("Bad epochs were dropped although "
                              "there shouldn't have been bad epochs."))

def checkEvents(events, expectedValues, raw, outpath=None, **kwargs):
    """
    Check whether the extracted events and the programmed events make sense.
    Add option to run some simple corrections of errors, if they are
    straightforward
    """
    uniques, count = np.unique(events[:, 2], return_counts=1)
    event_count = dict(zip(uniques, count))
    for k, v in expectedValues.items():
        if v is None:
            continue
        print(v, event_count[int(k)])
        if v != event_count[int(k)]:
            print(f'WARNING: NUMBER OF TRIGGERS FOR TRIGGER {k} DIFFERENT THAN'
                  ' EXPECTED!')

    # plot timing/number of events
    if outpath:
        fig = plot_events(events, sfreq=raw.info['sfreq'],
                                  first_samp=raw.first_samp, **kwargs)
        pt.save_plot(fig, outpath, dpi=600)


def checkErfPerRegion(ep_pre, ep_post, ch_type='mags', baseline=(-0.2, 0),
                      outpath='qc_erp_loc.png'):
    """
    For every NEUROMAG region of the scalp plots the ERPs of a set of epochs
    epochs:     EPOCh object | epochs to compute ERPs over
    ch_type:    str | grads or mags
    title_comment:  str extra comment to be added to the
    returns:    fig object
    """

    erp_pre = ep_pre.apply_baseline(baseline).average()
    erp_post = ep_post.apply_baseline(baseline).average()
    fig, ax = plt.subplots(8, 2, figsize=(15, 15))
    for erpI, (ID, erp) in enumerate(zip(['pre', 'post'],[erp_pre, erp_post])):
        for chI, chs in enumerate(['Left-frontal', 'Right-frontal',
                                   'Left-temporal', 'Right-temporal',
                                   'Left-parietal', 'Right-parietal',
                                   'Left-occipital', 'Right-occipital']):
            picks = getNeuromagRegions(erp, chs, exclude=True, ch_type=ch_type)
            erp.plot(show=False, proj=True, picks=picks, titles=f'{chs}_{ID}',
                     axes=ax[chI, erpI], spatial_colors=True)
    fig.suptitle(f'ERPs for each scalp region before (left) and after cleaning'
                 '(right)')
    pt.save_plot(fig, outpath=outpath, dpi=600)


def checkPSDPerRegion(ep_pre, ep_post, ch_type='mags', baseline=(-.5, -.3), 
                      outpath='qc_psd.png'):
    """
    For every NEUROMAG region of the scalp plots the PSD spectra
    of a set of epochs
    epochs:     EPOCh object | epochs to compute ERPs over
    ch_type:    str | grads or mags
    title_comment:  str extra comment to be added to the
    returns:    fig object
    """

    fig, ax = plt.subplots(8, 2, constrained_layout=True, figsize=(15, 15))
    for epI, (ID, ep) in enumerate(zip(['pre', 'post'],[ep_pre, ep_post])):
        ep.apply_baseline(baseline)
        for chI, chs in enumerate(['Left-frontal', 'Right-frontal',
                                   'Left-temporal', 'Right-temporal',
                                   'Left-parietal', 'Right-parietal',
                                   'Left-occipital', 'Right-occipital']):
            picks = getNeuromagRegions(ep, chs, exclude=True, ch_type=ch_type)
            ep.info['bads'] = [] 
            psd = ep.compute_psd(fmin=2, fmax=120)
            psd.plot(picks=picks, show=False, axes=ax[chI, epI],
                     spatial_colors=True)
            ax[chI, epI].set_title(f'{chs}_{ID}')
    fig.suptitle(f'PSDs for each scalp region before (left) and after cleaning'
                 '(right)')
    pt.save_plot(fig, outpath=outpath, dpi=600)


def classify_components(ica, raw, epochs, mode, deriv_stub=None,
                        qc_stub=None):
    """Find ica components linked to a certain type of artifact.""" 
    # find artifact-ICA matches
    if mode == 'veog':
        indices, scores = ica.find_bads_eog(raw, ch_name='EOG127',
                                            threshold=0.5, 
                                            measure='correlation')
    elif mode == 'heog':
        indices, scores = ica.find_bads_eog(raw, ch_name='EOG128',
                                            threshold=0.5,
                                            measure='correlation')
    elif mode == 'ecg':
        try:
            indices, scores = ica.find_bads_ecg(raw, threshold=0.5,
                                                measure='correlation')
        except TypeError as e:
            logging.info(("Due to bad ECG channel, no ECG events found."
                           "Remove ECG channel and try to estimate ECG") )
            raw.drop_channels('ECG063')
            indices, scores = ica.find_bads_ecg(raw, threshold=0.5,
                                                measure='correlation')

    ica.exclude = indices

    # barplot of ICA component scores
    fig = ica.plot_scores(scores, show=False, labels=mode)
    pt.save_plot(fig, dpi=300,
                outpath=op.join(qc_stub % f'ica_{mode}_scores.png'))

    # plot diagnostics
    for idx in indices:
        fig = ica.plot_properties(epochs, picks=[idx], show=False)[0]
        pt.save_plot(fig, outpath=op.join(qc_stub % f'ica_{idx}_{mode}_'
                                         'diagnose.png'), dpi=300)
    return indices


def plot_artifact(ica, raw, mode, indices, load_art, deriv_stub, qc_stub):
    # try to read the artifact evoked objects from file

    if load_art: 
         art_ev = read_evokeds(deriv_stub % f'{mode}_artifact-ave.fif')[0]
    else:
        if mode == 'veog':
            art_ev = create_eog_epochs(raw, ch_name='EOG127').average()
        elif mode == 'heog':
            art_ev = create_eog_epochs(raw, ch_name='EOG128').average()
        elif mode == 'ecg':
            art_ev = create_ecg_epochs(raw).average()
        
    write_evokeds(deriv_stub % f'{mode}_artifact-ave.fif', art_ev,
                  overwrite=True)

    art_ev.apply_baseline(baseline=(-0.2, 0))

    ica.exclude = indices

    # plot the actual artifact
    art_ev.apply_proj()
    if not np.isnan(art_ev.data).all():
        ica.exclude = indices
        # plot the actual artifact
        art_ev.apply_proj()
        for ch_type in ['grad', 'mag']:
            fig = art_ev.plot_joint(picks=ch_type, show=False)
            pt.save_plot(fig, dpi=300, 
                        outpath=qc_stub % f'ica_{mode}_artifact_{ch_type}.png')
        # plot sources pre and post
        fig = ica.plot_sources(art_ev, show=False)
        pt.save_plot(fig, outpath=qc_stub % f'ica_{mode}_source.png', dpi=300)

        # plot ICs applied to the evoked
        fig = ica.plot_overlay(art_ev, show = False)
        pt.save_plot(fig, outpath=qc_stub % f'ica_{mode}_corr.png', dpi=300)
    else:
        logging.info('No artifacts found!')


def compute_ER_SSP(er_path, n_grad=3, n_mag=5, outpath=None):
    """ Replace default projections with those based on empty room.

    er_path : String | PosixPath
        path to empty room recording
    n_grad : int
        number of components to extract from the gradiometers
    n_mag : int
        number of components to extract from the magnetometers
    outpath : String | PosixPath
       path of where the projections plots will be saved to
    """

    # load empty room and mark bad channels    
    er = mb.read_raw_bids(er_path)
    ax = er.plot_psd(average=True, spatial_colors=False,
                     dB=False, xscale='log', show=False)
    pt.save_plot(ax, outpath.format('absNoise'), dpi=600)
    er.plot(show=True, block=True, duration=60, n_channels=60)
    orig_projs = er.info['projs']

    # delete system projections
    er.del_proj()
    # compute SSPs
    er_projs = compute_proj_raw(er, n_grad=n_grad, n_mag=n_mag)
    
    if outpath:
        # visualize system projections
        fig = plot_projs_topomap(orig_projs, colorbar=True,
                                 vlim='joint', info=er.info, show=False)
        pt.save_plot(fig, outpath.format('proj_orig'), dpi=600)
        # visualize new projections
        fig = plot_projs_topomap(er_projs, colorbar=True,
                                 vlim='joint', info=er.info, show=False)
        pt.save_plot(fig, outpath.format('proj_ER'), dpi=600)
    return er_projs


def compute_ICA(epochs, AR_settings, l_freq=1, n_components=50,
                random_state=97, n_jobs=1, qc_stub=None):
    """ Convenience wrapper around running an ICA"""

    # highpass to avoid low drift noise
    epochs = epochs.load_data().filter(l_freq=l_freq, h_freq=None)

    if AR_settings['apply_ar'] == 'run':
        logging.info("Exclude noise in data prior to ICA")
        megs = pick_types(epochs.info, meg=True, eeg=False)
        rej, _ = run_autorej(epochs, megs, n_jobs, random_state)
        # rej.save(AR_settings['ar_path'], overwrite=True)
        epochs = rej.transform(epochs)
    else:
        logging.info("Step skipped: No noisy epochs marked")
    
    # run ICA
    ica = ICA(n_components=n_components, max_iter='auto', method='picard',
              random_state=random_state)
    ica.fit(epochs)

    logging.info("Fitting ICA finished. Plot some diagnostics\n")
    if qc_stub: 
        # plot components
        figs = ica.plot_components(show=False)
        for fI, fig in enumerate(figs, 1):
            pt.save_plot(fig, qc_stub % f'ica_components-{fI}.png',
                         dpi=300)
        
        # plot sources
        '''
        n_plots = int(np.ceil(n_components / 17))
        for idx in range(n_plots):
            start_idx = idx * 17
            stop_idx = min((idx + 1) * 17, n_components)
            picks = list(range(start_idx, stop_idx))
            fig = ica.plot_sources(epochs, picks=picks, show_scrollbars=False,
                                   show=False)
            pt.save_plot(fig, qc_stub % f'ica_sources-{idx}.png', dpi=600)
        '''
    return ica


def detect_blinks(raw, ch_name='EOG127', tmin=-0.2, tmax=0.2, event_id=1000,
                  ref_event='stim'):
    """Check whether blink occurred during a critical period, and mark it.
    
    Used to exclude trials for which blinks occurred during stimulus onset.

    raw : Raw Instance
        raw instance that is used to construct blink artifcats
    ch_name : string | list of strings
        Channels to use for blink detection
    tmin : int
        Time before blink (negative)
    tmax : int
        Time after blink
    event_id : int
        What new code to give to this type of event
    ref_event : String
        What event of the original raw is used to check for blink presence
    """
    # find EOG events and extract annotations
    eog_ep = create_eog_epochs(raw, ch_name=ch_name, tmin = -tmin, tmax=tmax,
                               event_id=event_id)
    eog_an = annotations_from_events(eog_ep.events, raw.info['sfreq'],
                                     event_desc={event_id:"BAD_blink"},
                                     orig_time=raw.annotations.orig_time)

    # adjust annotations to contain only the blink
    eog_an.onset = eog_an.onset + tmin
    eog_df = eog_an.to_data_frame()

    # extract raw annotations and onsets
    event_df = raw.annotations.to_data_frame()
    blink_onset = event_df.loc[event_df.description == ref_event, 'onset']

    ep_onset = eog_df.loc[:, 'onset']

    # find stim events that occur during the blink
    bad_epochs = []
    for elem in ep_onset: 
        diff = (blink_onset - elem).apply(timedelta.total_seconds)
        bad_epochs += (np.where(np.logical_and(diff > -(tmax - tmin),
                                               diff < 0))[0]).tolist()

    # not qc plots are produced. The EOG epochs are already defined in the ICA
    # at the end of the preprocessing, the final epochs can be checked whether
    # the blink epochs match up
    return bad_epochs


def extract_baseline_activity(ep, tmin, tmax):
    return ep.load_data().get_data(tmin=tmin, tmax=tmax).mean(axis=-1)


def filtering(raw, l_freq=None, h_freq=None, freqs=None, picks=None, fmin=0.1,
              fmax=250, filter_settings={}, outpath=None):
    """
    plot pre/post line noise filtering
    if l_freq, or h_freq defined, do low or high pass filtering
    if l_freq, and h_freq defined, do bandpass filtering
    if freq is defined, do notch filtering
    l_freq/h_freq and freqs are mutually exclusive
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 0])

    if freqs is None:
        logging.info("Do low/high/bandpass filtering")
        raw.filter(l_freq=l_freq, h_freq=h_freq, **filter_settings)
    else:
        logging.info("Do notch filtering")
        raw.notch_filter(freqs=freqs, picks=picks)

    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 1])

    fig.suptitle('Unfiltered (left) and filtered (right) raw power spectra')
    for axis in ax.flatten():
        if l_freq:
            axis.axvline(x=l_freq, ls='--', color='black')
        if h_freq:
            axis.axvline(x=h_freq, ls='--', color='black')

    pt.save_plot(fig, outpath, dpi=600)
    return raw


def find_bad_channels(raw, crosstalk_f, fine_cal_f, limit=3, duration=3,
                      min_count=20, h_freq=40, outpath=None):
    """finds bad channels based on maxwell stuff"""

    raw.info['bads'] = []

    # run SSS-based algorithm
    noisy, flat, scores = find_bad_channels_maxwell(raw,
        cross_talk=crosstalk_f, calibration=fine_cal_f,
        return_scores=True, verbose=True, limit=3, min_count=20,
        h_freq=h_freq, duration=3) # in theory h_freq=None should also be fine

    # extract values for plotting
    score = scores['scores_noisy']
    limits = scores['limits_noisy']
    dtp_limits = limits[:len(noisy)]
    bins = scores['bins']
    labels = [f'{start:3.0f} – {stop:3.0f}' for start, stop in bins]

    if outpath:
        # plot the bad segments
        # convert to pandas
        full = pd.DataFrame(score, columns=pd.Index(labels, name='Time (s)'),  
                            index=pd.Index(raw.ch_names[:306], name='Channel'))
        dtp = full.loc[noisy, :]

        fig, ax = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
        fig.suptitle(f'SSS BAD channels', fontsize=16, fontweight='bold')
        
        # First, plot the "raw" scores.
        heatmap(full, cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[0])
        ax[0].set_title('All Scores', fontweight='bold')

        # Now, adjust color range to highlight segments that exceeded the limit.
        heatmap(data=dtp, vmin=limit, cmap='Reds',
                    cbar_kws=dict(label='Score'), ax=ax[1])
        ax[1].set_title('Scores > Limit', fontweight='bold')

        # save figure
        pt.save_plot(fig, outpath, dpi=600)
    return noisy + flat


def fixEvents(events, target_set, rule='liberal'):
    """ Find weird events and fix them to what they are supposed to be

    If rule is conservative, no fixing is done, and weird events are simply
    excluded. If rule is interpolate, overlapping responses with feedback are
    interpolated. If rule is liberal, all events are fixed. Returns the cleaned
    events array.

    Default is liberal. Be careful to interpret the corrected values. Their
    meaningfulness can be compromised by the relative position (e.g. a
    response after a timeout has unique visual information, relative to a
    regular response). The starting_value of the events array can be used to
    identify modified events.
    """

    # first remove single-sample events
    single_samples = np.diff(events[:, 0]) > 1
    where_single = np.where(np.invert(single_samples))[0]
    if where_single.shape != (0,):
        events[where_single + 1, 1] = 0
        print(f'INFO: SINGLE-SAMPLE EVENTS {where_single} WERE DROPPED.')
        keep = np.array([True] * events.shape[0])
        keep[where_single] = False
        events = events[keep]

    # now look at unexpected events
    event_quality = np.isin(events[:, 2], target_set, invert=True)
    bad_events = np.where(event_quality)[0]
    undefined_events = []
    if rule == 'conservative':
        print(f'INFO: EVENTS {bad_events} WERE DROPPED.')
        return events[np.invert(event_quality)], undefined_events
    elif rule == 'liberal':
        events[bad_events, 2] = events[bad_events, 2] - events[bad_events, 1]
        for ev in events[bad_events, 2]:
            if ev not in target_set:
                undefined_events.append(ev)
                print(f'{ev} IS NOT A VALID TRIGGER VALUE. DOUBLECHECK!')
        print(f'INFO: VALUE OF EVENTS {bad_events} WERE CHANGED.')
        return events, undefined_events
    else:
        raise ValueError(f'{rule} is not a valid keyword for rule.')


def fix_bem_surfaces(subjects_dir, subject):
    import pymeshfix
    from mne.surface import read_surface 
    rr, tris = read_surface(op.join(subjects_dir, subject, 'surf',
                            'lh.seghead'))
    rr, tris = pymeshfix.clean_from_arrays(rr, tris)
    write_surface(op.join(subjects_dir, subject, 'surf', 'lh.seghead'),
                  rr, tris, overwrite=True)


def get_tSSS_duration(raw, low=10, high=19, step=0.5):
    """Based on duration of recording, compute optimal value for st_duration
    
    See MNE documentation about tSSS. If splitting up a recording into chunks, 
    all chunks should have the same length. This is not always given, so we 
    choose a size in the range of 10-19 s, that minimizes the rest, and 
    corresponds to a highpass filter of 0.05-0.1 Hz. 
    
    raw : Raw instance
        path to empty room recording
    low : int | float
        lowest permissible value for st_duration
    high : int | float
        highest permissible value for st_duration
    step : int | float
        step size for building the range of possible st_duration values
    """
    duration = int(raw.n_times / 1000)
    candidates = np.arange(low, high, step)
    winner = candidates[np.argmin(duration%candidates)]
    rest = duration % winner
    logging.info(f'tSSS split size is {winner}, meaning a tail of {rest} s.')
    return winner


def getNeuromagRegions(meg, label, exclude=False, ch_type='both'):
    """
    returns the indices of channels in a certain region of the Neuromag sensor
    space
    meg: Raw, Epoch, Evoked (or anything that has the "ch_names" field)
    label: list, str which region to extract. Possible fields are:
    Left-frontal, Right-frontal, Left-temporal, Right-temporal, Left-parietal,
    Right-parietal, Left-occipital, Right-occipital
    type: grads, mags or both
    returns a sorted list of indices that match the provided label
    """
    if exclude:
        bads = meg.info['bads']
    else:
        bads = []

    if isinstance(label, str):
        label = [label]
    if ch_type == 'mags':
        p = re.compile('MEG\d\d\d1')
    elif ch_type == 'grads':
        p = re.compile('MEG\d\d\d[2-3]')
    else:
        p = re.compile('MEG\d\d\d[1-3]')

    # extract channel labels from meg object
    sel = set()
    for l in label:
        sel = sel.union({ch.replace('MEG ', 'MEG')
                         for ch in read_vectorview_selection(l)})
    sel = list(sel)

    # use regexp to find only grads or mags
    channels = re.findall(p, '|'.join(sel))
    
    # get indices
    labelIdx = []
    for ch in channels:
        if not ch in bads and ch in meg.ch_names:
            labelIdx.append(meg.ch_names.index(ch))
    return sorted(labelIdx)


def interpolate_autoreject(epochs, ar_obj):
    """Does autoreject based interpolation """

    log_obj = ar_obj.get_reject_log(epochs)
    # apply interpolation
    ar.autoreject._apply_interp(log_obj, epochs, ar_obj.threshes_,
                                ar_obj.picks_, ar_obj.dots, ar_obj.verbose)
    return epochs


def filter_events(events, event_dict, label, pre_trg=None, post_trg=None,
                  pre_nontrg=None, post_nontrg=None):
    """From all events select a subset based on label

    To simplify code, label needs to be an interable!

    Function is quite idiosyncratic at the moment, specifically decided for HHU 
    MEG (key press at any time possible), for the RDM task (see stim
    trigger values). Perhaps we fix it at another time. 

    """
    if not isinstance(label, (list, tuple, np.ndarray)):
        raise ValueError('label must be a list or array or other sequence!')

    trigger = []
    for trg in label: 
        trigger.append(event_dict[trg])
    ep_id = {k:v for (k, v) in event_dict.items() if v in trigger}

    keep_idx = np.where(np.isin(events[:, 2], trigger))[0]
    
    if pre_trg is not None :
        priors = keep_idx - 1
        priors = priors[np.isin(events[priors, 2], pre_trg),]
        keep_idx = priors + 1

    if post_trg is not None :
        posts = keep_idx + 1
        posts = posts[np.isin(events[posts, 2], post_trg),]
        keep_idx = posts - 1

    keep_events = events[keep_idx]
    return keep_events.astype(int), ep_id


def generate_resplocked_epochs(ep, events, tmin=-1, tmax=0.8, timeout_trg=16):
    """Function to generate response locked epochs through time
    
    shifts from stim-locked epochs. Hardcoding here is not ideal """
    new_epochs = []
    resp_ep = ep['resp_time > 0.205'].load_data()
    for epI in range(resp_ep.events.shape[0]):
        epoch = resp_ep[epI]
        ev = epoch.events[:, 0]
        idx = np.where(events[:, 0] == ev)[0]
        if events[idx + 1, 2] == timeout_trg:
            logging.info("Dropping a timeout trial, bc there is no response")
            continue
        elif events[idx + 1, 2] not in [512, 2048]:
            logging.error("Problem with response epochs making!")
            raise ValueError(f"Problem with response epochs making! For {epI}")
        resp_time = events[idx + 1, 0]
        t_shift = 0.001 * (resp_time - ev)
        new_epochs.append(epoch.shift_time(-t_shift).crop(tmin=tmin, tmax=tmax))
    logging.info((f"There are {len(new_epochs)} resp-locked epochs," 
                  f" based on {ep.events.shape[0]} stim-locked epochs"))
    return concatenate_epochs(new_epochs)


def make_notch_filtered_epochs(raw, events, event_id, tmin, tmax, picks,
                               baseline, freq=10, notch_widths=2.5):
    raw_notch = raw.copy().load_data().notch_filter(freq,
                                                    notch_widths=notch_widths)
    return Epochs(raw_notch, events=events, event_id=event_id, tmin=tmin,
                  tmax=tmax, picks=picks, baseline=baseline)


def read_events(raw, bids_path):
    """Reads events in a robust-ish way. 

    Some subjects have weird events happening (double key presses, keypress
    during stimulus onset, etc.). This function is based on manual inspection
    and fixes erroneous events. Either by removing them or recoding them.
    """    

    # load event tsv
    sub = bids_path.subject
    ses = bids_path.session
    event_df = read_csv(bids_path.fpath, sep='\t')

    # extract unique events
    unique_events = event_df.value.unique()
    unique_ids = event_df.trial_type.unique()
    event_dict = {k:v for (k,v) in zip(unique_ids,unique_events)}

    # use event tsv sidecar to extract up to date events
    onsets = np.array(
        [np.nan if on == "n/a" else on for on in event_df["onset"]], dtype=float
    ) 
    # we need to add the first sample time, to align the other annotations below
    # with the events
    onsets += raw.first_time
    durations = np.array(
        [0 if du == "n/a" else du for du in event_df["duration"]], dtype=float
    )
    descriptions = np.asarray(event_df["trial_type"], dtype=str)
    annot_from_events = Annotations(onset=onsets, duration=durations,
                                    description=descriptions,
                                    orig_time=raw.annotations.orig_time)

    # now we need to add the annotated breaks to these annotations, 
    # otherwise they miss. So we extract them from the raw file
    annot_idx_to_keep = [
        idx for idx, descr in enumerate(raw.annotations.description)
        if descr == "BAD_break"
    ]
    annot_to_keep = raw.annotations[annot_idx_to_keep]
    raw.set_annotations(annot_from_events + annot_to_keep)

    events, event_dict = events_from_annotations(raw, event_dict)

    return events, event_dict


def load_misses(epochs):
    """Based on behav task/session return index of bad responses

    Assumes bids file structure, response coded as "resp_key", and
    a complete dataset
    """    

    # how many trials should be in a recording?
    if path.task == 'vbm':
        n_trials = 500
    elif path.task == 'ml':
        n_trials = 400
    elif path.task == 'rdm':
        n_trials = 826
    
    # read data
    path.datatype = 'beh'
    beh_path = path.fpath
    df = read_csv(str(beh_path) + '_beh.tsv', sep='\t')

    # if some trials are missing (because some idiot forgot to record on time)
    if epochs.events.shape[0] != n_trials:
        """idea:
        if mismatch of actual and planned #events, check the ses_info file for
        the combo of sub/ses/task. From this file, we can extract the indices of
        the events that were missing. Those events can then be removed from the 
        dataframe, so that the indices match up again. 
        TODO: 
        - make that file (requires manual inspection of the data file)
        - implement the procedure, subjectwise
        """
        raise ValueError("Data file is not complete! Check data manually!")

    # otherwise return the indices of the missed responses
    return list(df.loc[pd.isnull(df.resp_key), :].index)


def maxwell_filter_wrapper(raw, calibration, cross_talk, duration, correlation,
                   outpath=None):

    # pre filter power spectrum
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 0])

    # run maxwell filtering algorithm
    raw = maxwell_filter(raw, verbose=True, calibration=calibration,
                            cross_talk=cross_talk, st_duration=duration,
                            st_correlation=correlation)
    
    # post filter power spectrum
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 1])

    fig.suptitle('Unprocessed (left) and processed (right) raw power spectra')
    pt.save_plot(fig, outpath, dpi=600)
    return raw


def oversampled_temporal_projection_wrapper(raw, outpath=None):

    # pre filter power spectrum
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 0])

    # run oversampled_temporal_projection
    raw = oversampled_temporal_projection(raw)
    
    # post filter power spectrum
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 1])

    fig.suptitle('Unprocessed (left) and processed (right) raw power spectra')
    pt.save_plot(fig, outpath, dpi=600)
    return raw


def plot_diagnostic_ERF(dirty_ep, clean_ep, filter, baseline=(-0.2, 0),
                        outpath=None):
    """some simple ERF comparison plot"""

    l_oc_grad = getNeuromagRegions(dirty_ep, ['Left-occipital'], exclude=True,
                                   ch_type='grads')
    l_oc_mag = getNeuromagRegions(dirty_ep, ['Left-occipital'], exclude=True,
                                  ch_type='mags')
    r_oc_grad = getNeuromagRegions(dirty_ep, ['Right-occipital'], exclude=True,
                                   ch_type='grads')
    r_oc_mag = getNeuromagRegions(dirty_ep, ['Right-occipital'], exclude=True,
                                  ch_type='mags')
    comparisons = [[l_oc_grad, r_oc_grad], [l_oc_mag, r_oc_mag]]

    dirty_ev = dirty_ep.average().apply_baseline(baseline = baseline)
    clean_ev = clean_ep.average().apply_baseline(baseline = baseline)
    stim_times = [-0.2,0,0.1,0.2,0.4,1]
    title_id = [['Gradiometers - dirty','Gradiometers - clean'],
                ['Magnetometers - dirty', 'Magnetometers - clean']]

    # all ERPs   
    fig, ax = plt.subplots(2, 2, figsize=(12, 12), dpi=600)

    for evI, ev in enumerate([dirty_ev, clean_ev]):
        for axI in range(2):
            ax[evI, axI].tick_params(axis='x', which='both', top='off')
            ax[evI, axI].tick_params(axis='y', which='both', right='off')
            picks = pick_types(ev.info, meg=['grad', 'mag'][axI], 
                               stim=False, eog=False, exclude=[])
            ev.plot(picks, exclude=[], axes=ax[evI, axI], show=False)
            ax[evI, axI].axhline(0, linewidth=.6, linestyle='--', color='black')
            ax[evI, axI].axvline(0, linewidth=.6, linestyle='--', color='black')
            ax[evI, axI].set_title(title_id[axI][evI])
            ax[evI, axI].set_ylabel('Amplitude (MEG unit)')
            ax[evI, axI].set_xlabel('Time (s)')
    plt.tight_layout()
    pt.save_plot(fig, outpath=outpath.replace('erf', 'erf_allCh'), dpi=600)

    # some visual ERP 
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 5),
                           sharex=True)

    for evI, ev in enumerate([dirty_ev, clean_ev]):
        for cI, (c1, c2) in enumerate(comparisons):
            ev = ev.filter(l_freq=filter[0], h_freq=filter[1])
            ev1 = ev.data[c1, :].mean(axis=0).squeeze()
            ev2 = ev.data[c2, :].mean(axis=0).squeeze()

            ax[cI, evI].plot(ev.times, ev1, ev.times, ev2)
            ax[cI, evI].set_xlabel('Time (s)')
            ax[cI, evI].set_ylabel('Amplitude (MEG unit)')
            ax[cI, evI].axhline(0, linewidth=0.6, linestyle='--', color='black')
            ax[cI, evI].axvline(0, linewidth=0.6, linestyle='--', color='black')
            ax[cI, evI].set_title(f'Lateralized OCC ERF - {title_id[cI][evI]}')
    pt.save_plot(fig, outpath, dpi=600)

    for evI, ev in enumerate([dirty_ev, clean_ev]):
        preproc = ['dirty', 'clean'][evI]
        fig_topo = ev.plot_topomap(times=stim_times, ch_type='grad',
                                    colorbar = True, show=False)
        pt.save_plot(fig_topo, outpath.replace('erf_prepost', 
                                              f'erf_grad_topo_{preproc}'),
                    dpi=600)
        fig_topo = ev.plot_topomap(times=stim_times, ch_type='mag',
                                    colorbar = True, show=False)
        pt.save_plot(fig_topo, outpath.replace('erf_prepost',
                                              f'erf_mag_topo_{preproc}'),
                    dpi=600)


def plot_diagnostic_TF(dirty_ep, clean_ep, outpath, baseline=(-.5, -.3)):
    """Simple diagnostic timefrequency plot"""
    freqs = np.logspace(*np.log10([4, 100]), num=20)
    n_cycles = freqs / 2. 

    title_id = [['Gradiometers - dirty','Gradiometers - clean'],
                ['Magnetometers - dirty', 'Magnetometers - clean']]

    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharey=True)

    for epI, ep in enumerate([dirty_ep, clean_ep]):
        preproc = ['dirty', 'clean'][epI]
        ep.apply_baseline(baseline)

        power = tfr_morlet(ep, use_fft=True, freqs=freqs,
                                         n_cycles=n_cycles, return_itc=False)
        power = power.crop(-0.5,1.6) 

        oc_grad = getNeuromagRegions(power, ['Left-occipital', 'Right-occipital'], exclude=True, ch_type='grads')
        oc_mag = getNeuromagRegions(power, ['Left-occipital', 'Right-occipital'], exclude=True, ch_type='mags')

        power.plot(oc_grad, baseline=(-0.5, -0.3), combine='mean', 
                   mode='logratio', axes=ax[0, epI], show=False)
        power.plot(oc_mag, baseline=(-0.5, -0.3), combine='mean',
                   mode='logratio', axes=ax[1, epI], show=False)
        ax[0, epI].set_title(title_id[0][epI])
        ax[1, epI].set_title(title_id[1][epI])

        # also do topoplot
        fig_topo, ax_topo = plt.subplots(2, 2, figsize=(7.5, 3.75),
                                         sharey=True)
        i = 0 
        for freq, (fmin, fmax) in zip(['Theta', 'Alpha', 'Beta', 'Gamma'],
                                      [[5, 8], [8, 12], [13, 30], [30, 90]]):
            power.plot_topomap(ch_type='grad', fmin=fmin, fmax=fmax,
                               mode='logratio', axes=ax_topo[i // 2, i % 2],
                               title=freq, show=False)
            i += 1
        pt.save_plot(fig_topo, outpath.replace('tf_prepost',
                                              f'tf_topo_{preproc}'),
                    dpi=600)
    pt.save_plot(fig, outpath, dpi=600)



def run_autorej(epochs, picks, n_jobs=1, seed=10, outpath=None):
    """Use autoreject (local) to find bad and maybe fix epochs"""

    # set parameters for autoreject
    n_interpolates = np.array((1, 2, 3, 5, 7, 9))
    consensus_percs = np.linspace(0.2, 0.7, 11)


    # init autoreject object
    rej = ar.AutoReject(n_interpolates, consensus_percs, picks=picks,
                        thresh_method='bayesian_optimization',
                        random_state=seed, n_jobs=n_jobs, verbose=2)

    # run the autoreject (local) algorithm
    rej.fit(epochs)

    # extract the log file
    rej_log = rej.get_reject_log(epochs)

    # plot the results
    if outpath is not None:
        viz_autorej(rej_log, outpath=outpath)

    # return it 
    return rej, rej_log


def set_bad_channels(channel_infile):
    try:
        channels_df = pd.read_csv(channel_infile, sep='\t')
    except FileNotFoundError as e:
        logging.error(f'{channel_infile} does not exist.')
    return channels_df.loc[channels_df.status == 'bad', 'name'].tolist()


def viz_autorej(reject_log, outpath):
    """
    Solutions for both gradiometers and magnetometers are combined here. 
    Exclusion of bad channels will be done later. Here only the bad epoch
    indices are noted

    reject_log:     RejectLog
        Log object as produced by autoreject
    outpath:    str, Path
        path to save the qc plot to
    """

    # plot channel x epoch image with info on good, bad, interpolation
    fig, axes = plt.subplots(1, 1, figsize=(18, 24), dpi=1200)    
    # prep
    xlabels = reject_log.ch_names
    image = reject_log.labels
    image[image == 2] = 0.5  # move interp to 0.5
    legend_label = {0: 'good', 0.5: 'interpolated', 1: 'bad'}
    cmap = colors.ListedColormap(['white', 'blue', 'red'])
    # plot
    img = axes.imshow(image.T, cmap=cmap, vmin=0, vmax=1, interpolation='none')
    axes.set(xlabel='Epochs', ylabel='Channels')
    plt.setp(axes, yticks=range(0, len(xlabels)), yticklabels=xlabels)
    plt.setp(axes.get_yticklabels(), fontsize=2)
    #add red box around rejected epochs
    for idx in np.where(reject_log.bad_epochs)[0]:
        axes.add_patch(patches.Rectangle((idx - 0.5, -0.5), 1, len(xlabels),
                     linewidth=1, edgecolor='r', facecolor='none'))
    # add legend
    handles = []
    for i, label in legend_label.items():
        handles.append(patches.Patch(color=img.cmap(img.norm(i)), label=label))
    axes.legend(handles=handles, bbox_to_anchor=(3.5, 0.5), ncol=1,
                      borderaxespad=0.)
    pt.save_plot(fig, outpath=outpath, dpi=600)


def zapline(raw, line_freq=50, sample_freq=1000, nremove=10, outpath=None):

    from meegkit.dss import dss_line

    # pre filter power spectrum
    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True, sharey=True)
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 0])

    mag_picks = pick_types(raw.info, meg='mag', eeg=False)
    planar1_picks = pick_types(raw.info, meg='planar1', eeg=False)
    planar2_picks = pick_types(raw.info, meg='planar2', eeg=False)

    # do every channel type one by one to save memory
    data = raw.get_data(picks=mag_picks)
    raw._data[mag_picks] = dss_line(data.T, fline=line_freq, sfreq=sample_freq,
                                nremove=nremove)[0].T
    data = raw.get_data(picks=planar1_picks)
    raw._data[planar1_picks] = dss_line(data.T, fline=line_freq, 
                                        sfreq=sample_freq,
                                        nremove=nremove)[0].T    
    data = raw.get_data(picks=planar2_picks)
    raw._data[planar2_picks] = dss_line(data.T, fline=line_freq, 
                                        sfreq=sample_freq,
                                        nremove=nremove)[0].T
    # post filter power spectrum
    psd = raw.compute_psd()
    psd.plot(xscale='log', average=False, show=False, axes=ax[:2, 1])

    fig.suptitle('Unfiltered (left) and filtered (right) raw power spectra')
    pt.save_plot(fig, outpath, dpi=600)
    return raw
