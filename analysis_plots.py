import cPickle as pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from Chandra.Time import DateTime

import pyyaks.logger

logger = pyyaks.logger.get_logger(format='%(asctime)s: %(message)s')


def plot_obsid(obsid, sp=None, dp=None, ir=None, ms=None, anyflag=None, slots=None):
    """
    The value of 3.0 was semi-empirically derived as the value which minimizes
    the centroid spreads for a few obsids.  It also corresponds roughly to
    2.05 + (2.05 - 1.7 / 2) which could be the center of the ACA integration.
    Some obsids seem to prefer 2.0, others 3.0.
    """
    filename = os.path.join('data', str(obsid) + '.pkl')
    if os.path.exists(filename):
        dat = pickle.load(open(filename, 'r'))
    else:
        import update_flags_archive
        dat, telems = update_flags_archive.get_obsid(obsid)
        pickle.dump(dat, open(filename, 'w'), protocol=-1)
        logger.info('Wrote data for {}'.format(obsid))

    plt.clf()
    for slot in slots or dat['slots']:
        dyag = dat['dyag'][slot]
        dzag = dat['dzag'][slot]
        ok = np.ones(len(dyag), dtype=bool)
        flag_vals = {'sp': sp, 'ir': ir, 'ms': ms, 'dp': dp}
        for flag in flag_vals:
            if flag_vals[flag] is not None:
                msid = 'aoaci{}'.format(flag)
                match_val = (1 if flag_vals[flag] else 0)
                ok &= dat['vals'][msid][slot] == match_val
        if np.any(ok):
            times = dat['times']['aoacyan'][slot][ok]
            dyag = dyag[ok]
            dzag = dzag[ok]
            plt.plot(times, dzag, '.', ms=2.0)
            p16, p84 = np.percentile(dyag, [15.87, 84.13])
            y_sig = (p84 - p16) / 2
            y_std = np.std(dyag)
            p16, p84 = np.percentile(dzag, [15.87, 84.13])
            z_sig = (p84 - p16) / 2
            z_std = np.std(dzag)
            logger.info('Slot {}: {} values: y_sig={:.2f} y_std={:.2f} z_sig={:.2f} z_std={:.2f}'
                        .format(slot, np.sum(ok), y_sig, y_std, z_sig, z_std))
        else:
            logger.info('Slot {}: no data values selected')
    plt.grid()
    plt.ylim(-5.0, 5.0)
    plt.title('Obsid {} at {}'.format(obsid, DateTime(dat['time0']).date[:17]))
    plt.show()
