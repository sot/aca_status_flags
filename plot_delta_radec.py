
import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
import numpy as np
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


slots = range(0,1)
slot_msids = [ msid + "%s" % slot for slot in slots for msid in pcad_msids ]

msids = ['aopcadmd',
         'aoattqt1',
         'aoattqt2',
         'aoattqt3',
         'aoattqt4',
         ]

msids.extend(slot_msids)

obsid = '56345'

db = Ska.DBI.DBI(dbi='sybase',server='sybase')
if 'obs' not in globals():
    obs = db.fetchall("""select obsid, kalman_datestart, kalman_datestop
    from observations where obsid = %s""" % obsid)[0]

if 'telem' not in globals():
    telem = fetch.MSIDset( msids,
                           obs['kalman_datestart'],
                           obs['kalman_datestop'],
                           filter_bad=True,
                           )

vals = [telem['aoattqt%d' % i].vals for i in range(1,5)] + [telem['aoacyan0'].vals/3600.,
                                                       telem['aoaczan0'].vals/3600.]
if 'coords' not in globals():
    radecs = [yagzag2radec(yag, zag, Quat([q1, q2, q3, q4]))
              for q1, q2, q3, q4, yag, zag in zip(*vals)]
    coords = np.rec.fromrecords(radecs, names=('ra', 'dec'))

    # aborted code to transform back to dyag, dzag
    if 0:  
        radecrolls = [Quat([q1, q2, q3, q4]).equatorial 
                      for q1, q2, q3, q4, yag, zag in zip(*vals)]
        radecrolls = np.rec.fromrecords(radecrolls, names=('ra', 'dec', 'roll'))
        ra_nom = np.mean(radecrolls['ra'])
        dec_nom = np.mean(radecrolls['dec'])
        roll_nom = np.mean(radecrolls['roll'])
        q_nom = Quat([ra_nom, dec_nom, roll_nom])

ok = telem['aoacfct0'].vals == 'TRAK'
slot = 0

flags = { 'dp' : telem['aoacidp%s' % slot].vals != 'OK ',
          'ir' : telem['aoaciir%s' % slot].vals != 'OK ',
          'ms' : telem['aoacims%s' % slot].vals != 'OK ',
          'sp' : telem['aoacisp%s' % slot].vals != 'OK ',
          }

color = { 'dp' : 'blue',
          'ir' : 'red',
          'ms' : 'green',
          'sp' : 'orange'
          }

def plot_axis(label, dy, filename=None):
    figure(figsize=(5, 5))
    subplot(2, 1, 1)
    clean = ok & ~flags['dp'] & ~flags['ir'] & ~flags['ms'] & ~flags['sp']
    plot_cxctime(times[clean], dy[clean], 'k.', markersize=2)
    ylabel('Delta %s (arcsec)' % label)
    ylim(-10,10)

    subplot(2, 1, 2)
    plot_cxctime(times[ok], dy[ok], 'k.', markersize=2)
    for flag in flags:
         bad = ok & flags[flag]
         if any(bad):
             plot_cxctime(times[bad], dy[bad],
                          color=color[flag],
                           markersize=3, marker='.',
                           linestyle='None')
    ylabel('Delta %s (arcsec)' % label)
    ylim(-10,10)

    if filename is not None:
        savefig(filename)

times = telem['aoacyan%s' % slot].times
dra = (coords['ra'] - np.mean(coords['ra'][ok])) * 3600 * np.cos(np.radians(coords['dec']))
ddec = (coords['dec'] - np.mean(coords['dec'][ok])) * 3600

plot_axis('RA', dra, 'delta_ra.png')
plot_axis('Dec', ddec, 'delta_dec.png')
#plot_axis('RA', dra, None)
#plot_axis('Dec', ddec, None)



