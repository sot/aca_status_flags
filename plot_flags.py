import numpy as np
import matplotlib.pyplot as plt

import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
from Quaternion import Quat
from quatutil import yagzag2radec
from Chandra.Time import DateTime

pcad_msids = ['aoacfct',
              'aoacicc',
              'aoacidp',
              'aoaciir',
              'aoacims',
              'aoaciqb',
              'aoacisp',
              'aoacyan',
              'aoaczan',
              'aoimage']


def plot_centroid_resids_by_flag(start, stop, slot, save=False):
    slot_msids = [msid + "%s" % slot for msid in pcad_msids]

    msids = ['aopcadmd',
             'aoattqt1',
             'aoattqt2',
             'aoattqt3',
             'aoattqt4',
             ]

    msids.extend(slot_msids)

    print('Fetching telemetry from {} to {}'.format(start, stop))
    telem = fetch.MSIDset(msids,
                          start,
                          stop,
                          filter_bad=True,
                          )
    telem.interpolate(dt=2.05)

    vals = ([telem['aoattqt%d' % i].vals for i in range(1, 5)]
            + [telem['aoacyan{}'.format(slot)].vals / 3600.,
               telem['aoaczan{}'.format(slot)].vals / 3600.])

    print('Interpolating quaternions')
    radecs = [yagzag2radec(yag, zag, Quat([q1, q2, q3, q4]))
              for q1, q2, q3, q4, yag, zag in zip(*vals)]
    coords = np.rec.fromrecords(radecs, names=('ra', 'dec'))

    ok = telem['aoacfct{}'.format(slot)].vals == 'TRAK'

    flags = {'dp': telem['aoacidp%s' % slot].vals != 'OK ',
             'ir': telem['aoaciir%s' % slot].vals != 'OK ',
             'ms': telem['aoacims%s' % slot].vals != 'OK ',
             'sp': telem['aoacisp%s' % slot].vals != 'OK ',
             }

    times = telem['aoacyan%s' % slot].times
    dra = (coords['ra'] - np.mean(coords['ra'][ok])) * 3600 * np.cos(np.radians(coords['dec']))
    ddec = (coords['dec'] - np.mean(coords['dec'][ok])) * 3600
    dr = np.sqrt(dra ** 2 + ddec ** 2)

    fileroot = 'flags_{}'.format(DateTime(start).date[:14]) if save else None

    print('Making plots with output fileroot={}'.format(fileroot))
    for dp in (False, True):
        for ir in (False, True):
            for ms in (False, True):
                for sp in (False, True):
                    print('Making plot for dp={} ir={} ms={} sp={}'.format(dp, ir, ms, sp))
                    plot_axis('dR', times, dr, ok, dp, ir, ms, sp, flags, filename=fileroot)


def plot_axis(label, times, dy, ok, dp, ir, ms, sp, flags, filename=None):
    plt.figure(figsize=(6, 4))
    plt.clf()
    filt = (ok & (flags['dp'] == dp) & (flags['ir'] == ir)
            & (flags['ms'] == ms) & (flags['sp'] == sp))

    plot_cxctime(times[ok], dy[ok], 'r.', markersize=0.3)
    if len(np.flatnonzero(filt)) > 0:
        plot_cxctime(times[filt], dy[filt], 'k.', markersize=2)
    plt.ylabel('Delta %s (arcsec)' % label)
    plt.title('DP={0} IR={1} MS={2} SP={3}'.format(dp, ir, ms, sp))
    plt.ylim(-10, 10)
    plt.subplots_adjust(bottom=0.05, top=0.85)

    if filename is not None:
        ext = str(dp)[0] + str(ir)[0] + str(ms)[0] + str(sp)[0]
        plt.savefig(filename + ext + '.png')
