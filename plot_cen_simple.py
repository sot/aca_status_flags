
import Ska.engarchive.fetch as fetch
from Ska.Matplotlib import plot_cxctime
import numpy as np
import Quaternion

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


slots = range(0, 8)
slot_msids = [msid + "%s" % slot for slot in slots for msid in pcad_msids]

msids = ['aopcadmd',
         'aoattqt1',
         'aoattqt2',
         'aoattqt3',
         'aoattqt4',
         ]

msids.extend(slot_msids)


obsid = '56345'

import Ska.DBI
db = Ska.DBI.DBI(dbi='sybase', server='sybase')
if 'obs' not in globals():
    obs = db.fetchall("""select obsid, kalman_datestart, kalman_datestop
    from observations where obsid = %s""" % obsid)[0]

if 'telem' not in globals():
    telem = fetch.MSIDset(msids,
                          obs['kalman_datestart'],
                          obs['kalman_datestop'],
                          filter_bad=True,
                          )

dqs = []
quat0 = Quaternion.Quat([telem['aoattqt1'].vals[0],
                         telem['aoattqt2'].vals[0],
                         telem['aoattqt3'].vals[0],
                         telem['aoattqt4'].vals[0],
                         ])

for qidx in range(0, len(telem['aoattqt1'].vals)):
    quat = Quaternion.Quat([telem['aoattqt1'].vals[qidx],
                            telem['aoattqt2'].vals[qidx],
                            telem['aoattqt3'].vals[qidx],
                            telem['aoattqt4'].vals[qidx],
                            ])
    dq = quat0 / quat
    dqs.append(dq.q)

dqs = np.array(dqs)
r2a = 3600. * 180. / pi

clip = 1000
figsize = (6, 3)
for slot in slots[:1]:
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

    fig = figure(1, figsize=(5, 7))
    ax1 = subplot(411)

    ytimes = telem['aoacyan%s' % slot].times
    yang = telem['aoacyan%s' % slot].vals
    yok = abs(yang - yang.mean()) < clip

    plot_cxctime(ytimes[yok], yang[yok] - dqs[yok, 2] * r2a * 2, 'k.', markersize=1)
    for flag in flags.keys():
        bad = yok & flags[flag]
        if any(bad):
            plot_cxctime(ytimes[bad], yang[bad] - dqs[bad, 2] * r2a * 2,
                         color=color[flag],
                         markersize=3, marker='.',
                         linestyle='None')
    ylabel('yang')
    setp(ax1, xticklabels=[])

    ax1 = subplot(412)
    plot_cxctime(ytimes[yok], yang[yok] - dqs[yok, 2] * r2a * 2, 'k.', markersize=1)
    ylabel('yang')
    setp(ax1, xticklabels=[])

    ax2 = subplot(413)
    ztimes = telem['aoaczan%s' % slot].times
    zang = telem['aoaczan%s' % slot].vals
    zok = abs(zang - zang.mean()) < clip
    plot_cxctime(ztimes[zok], zang[zok] - dqs[yok, 1] * r2a * 2, 'k.', markersize=1)
    for flag in flags.keys():
        bad = zok & flags[flag]
        if any(bad):
            plot_cxctime(ztimes[bad], zang[bad] - dqs[bad, 1] * r2a * 2,
                         color=color[flag],
                         markersize=3, marker='.',
                         linestyle='None')
    ylabel('zang')

    ax2 = subplot(414)
    plot_cxctime(ztimes[zok], zang[zok] - dqs[yok, 1] * r2a * 2, 'k.', markersize=1)
    ylabel('zang')
    suptitle('Slot %s' % slot)

    savefig('centroids_fc_slot%s.png' % slot)

    if 0:
        ytimes = telem['aoacyan%s' % slot].times
        yang = telem['aoacyan%s' % slot].vals
        yok = abs(yang - yang.mean()) < clip

        ztimes = telem['aoaczan%s' % slot].times
        zang = telem['aoaczan%s' % slot].vals
        zok = abs(zang - zang.mean()) < clip

        figure(figsize=figsize)
        ax1 = subplot(2, 1, 1)
        plot_cxctime(ytimes[yok],
                     yang[yok] - dqs[yok, 2] * r2a * 2, 'k.', markersize=1)

        ylabel('dyang')

        ax2 = subplot(2, 1, 2)
        plot_cxctime(ztimes[zok],
                     zang[zok] - dqs[zok, 1] * r2a * 2,
                     'k.', markersize=1)

        ylabel('dzang')
        suptitle('Slot %s' % slot)
