import numpy as np
import matplotlib.pyplot as plt

import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
from Quaternion import Quat
import Ska.DBI
from quatutil import yagzag2radec

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

slots = [6]
slot_msids = [msid + "%s" % slot for slot in slots for msid in pcad_msids]

msids = ['aopcadmd',
         'aoattqt1',
         'aoattqt2',
         'aoattqt3',
         'aoattqt4',
         ]

msids.extend(slot_msids)

obsid = 15304  # 95 second drop on 2013:104:23:40:08.  See also obsid 56345.
obsid = 15175
obsid = 56345

db = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read')
if 'obs' not in globals():
    obs = db.fetchall("""select obsid, kalman_datestart, kalman_datestop
    from observations where obsid = {}""".format(obsid))[0]

if 'telem' not in globals():
    telem = fetch.MSIDset(msids,
                          obs['kalman_datestart'],
                          obs['kalman_datestop'],
                          filter_bad=True,
                          )
telem.interpolate(dt=2.05)

slot = 6
vals = ([telem['aoattqt%d' % i].vals for i in range(1, 5)]
        + [telem['aoacyan{}'.format(slot)].vals / 3600.,
           telem['aoaczan{}'.format(slot)].vals / 3600.])
if 'coords' not in globals():
    radecs = [yagzag2radec(yag, zag, Quat([q1, q2, q3, q4]))
              for q1, q2, q3, q4, yag, zag in zip(*vals)]
    coords = np.rec.fromrecords(radecs, names=('ra', 'dec'))


ok = telem['aoacfct{}'.format(slot)].vals == 'TRAK'

flags = {'dp': telem['aoacidp%s' % slot].vals != 'OK ',
         'ir': telem['aoaciir%s' % slot].vals != 'OK ',
         'ms': telem['aoacims%s' % slot].vals != 'OK ',
         'sp': telem['aoacisp%s' % slot].vals != 'OK ',
         }

color = {'dp': 'blue',
         'ir': 'red',
         'ms': 'green',
         'sp': 'orange'
         }


def plot_axis(label, dy, ok, dp, ir, ms, sp, filename=None):
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

times = telem['aoacyan%s' % slot].times
dra = (coords['ra'] - np.mean(coords['ra'][ok])) * 3600 * np.cos(np.radians(coords['dec']))
ddec = (coords['dec'] - np.mean(coords['dec'][ok])) * 3600

for dp in (False, True):
    for ir in (False, True):
        for ms in (False, True):
            for sp in (False, True):
                plot_axis('RA', dra, ok, dp, ir, ms, sp, '{}/flags_ra_'.format(obsid))


# plot_axis('Dec', ddec, 'delta_dec.png')
# plot_axis('RA', dra, None)
# plot_axis('Dec', ddec, None)
